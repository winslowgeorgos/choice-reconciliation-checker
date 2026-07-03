# cache_manager.py
import streamlit as st
import pandas as pd
import json
import hashlib
import pickle
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import uuid

# Cache directories
CACHE_DIR = Path("data/cache")
MANIFEST_DIR = Path("data/manifests")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

HASH_TRACKING_FILE = CACHE_DIR / "hash_tracking.json"
MANIFEST_FILE = MANIFEST_DIR / "save_manifest.json"

# ============================================================================
# MODULE CONFIGURATION - Define which session state keys belong to which module
# ============================================================================

MODULE_CONFIG = {
    "fx_reconciliation": {
        "display_name": "FX Adjustments Reconciliation",
        "keys": [
            'matched_local', 'unmatched_local', 'matched_foreign', 'unmatched_foreign', 'bank_records',
            'moved_local_matched', 'moved_local_unmatched', 'moved_foreign_matched', 
            'moved_foreign_unmatched', 'moved_bank_records', 'audit_moves_log',
            'deleted_local_matched', 'deleted_local_unmatched', 'deleted_foreign_matched',
            'deleted_foreign_unmatched', 'deleted_bank_records', 'audit_deletes_log',
            'moved_stats', 'deleted_stats', 'df_matched_adjustments_local', 'df_unmatched_adjustments_local',
            'df_matched_adjustments_foreign', 'df_unmatched_adjustments_foreign', 'df_unmatched_bank_records'
        ],
        "pickle_files": [
            "matched_local.pkl", "unmatched_local.pkl", "matched_foreign.pkl", 
            "unmatched_foreign.pkl", "bank_records.pkl",
            "moved_local_matched.pkl", "moved_local_unmatched.pkl", 
            "moved_foreign_matched.pkl", "moved_foreign_unmatched.pkl",
            "moved_bank_records.pkl", "audit_moves_log.pkl",
            "deleted_local_matched.pkl", "deleted_local_unmatched.pkl",
            "deleted_foreign_matched.pkl", "deleted_foreign_unmatched.pkl",
            "deleted_bank_records.pkl", "audit_deletes_log.pkl",
            "df_matched_adjustments_local.pkl", "df_unmatched_adjustments_local.pkl",
            "df_matched_adjustments_foreign.pkl", "df_unmatched_adjustments_foreign.pkl",
            "df_unmatched_bank_records.pkl"
        ]
    },
    "fx_trade_reconciliation": {
        "display_name": "FX Trade Reconciliation",
        "keys": [
            'df_matched_counterparty', 'df_matched_choice', 
            'df_unmatched_counterparty', 'df_unmatched_choice', 
            'df_unmatched_bank_trade'
        ],
        "pickle_files": [
            "df_matched_counterparty.pkl", "df_matched_choice.pkl",
            "df_unmatched_counterparty.pkl", "df_unmatched_choice.pkl",
            "df_unmatched_bank_trade.pkl"
        ]
    },
    "intermediary_reconciliation": {
        "display_name": "Intermediary Reconciliation",
        "keys": [
            'df_matched_intermediary_credit', 'df_matched_intermediary_debit',
            'df_unmatched_intermediary_credit', 'df_unmatched_intermediary_debit',
            'df_unmatched_bank_intermediary'
        ],
        "pickle_files": [
            "df_matched_intermediary_credit.pkl", "df_matched_intermediary_debit.pkl",
            "df_unmatched_intermediary_credit.pkl", "df_unmatched_intermediary_debit.pkl",
            "df_unmatched_bank_intermediary.pkl"
        ]
    },
    "interfund_reconciliation": {
        "display_name": "Interfund Reconciliation",
        "keys": [
            'df_matched_interfund', 'df_unmatched_interfund'
        ],
        "pickle_files": [
            "df_matched_interfund.pkl", "df_unmatched_interfund.pkl"
        ]
    },
    "business_fx_reconciliation": {
        "display_name": "Business FX Reconciliation",
        "keys": [
            'df_business_matched', 'df_business_unmatched', 'business_analysis_results'
        ],
        "pickle_files": [
            "df_business_matched.pkl", "df_business_unmatched.pkl", "business_analysis_results.pkl"
        ]
    },
    "cross_match_analysis": {
        "display_name": "Cross-Match Analysis",
        "keys": [
            'cross_match_results', 'cross_match_summary'
        ],
        "pickle_files": [
            "cross_match_results.pkl", "cross_match_summary.pkl"
        ]
    }
}

# ============================================================================
# HASH TRACKING - Single source of truth for change detection
# ============================================================================

def load_hash_tracking() -> Dict[str, str]:
    """Load the centralized hash tracking file"""
    if HASH_TRACKING_FILE.exists():
        try:
            with open(HASH_TRACKING_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_hash_tracking(hashes: Dict[str, str]):
    """Save the centralized hash tracking file"""
    try:
        # Convert non-serializable objects
        serializable_hashes = {}
        for key, value in hashes.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                serializable_hashes[key] = value
            else:
                serializable_hashes[key] = str(value)
        
        with open(HASH_TRACKING_FILE, 'w') as f:
            json.dump(serializable_hashes, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving hash tracking: {e}")
        return False

def compute_df_hash(df: pd.DataFrame) -> Optional[str]:
    """Compute a hash for a dataframe to detect changes"""
    if df is None or df.empty:
        return None
    try:
        # Remove row numbers and record IDs for hash calculation
        df_copy = df.copy()
        if '#' in df_copy.columns:
            df_copy = df_copy.drop(columns=['#'])
        if '_record_id' in df_copy.columns:
            df_copy = df_copy.drop(columns=['_record_id'])
        
        # Hash the dataframe content
        df_hash = pd.util.hash_pandas_object(df_copy, index=True).sum()
        return str(df_hash)
    except Exception as e:
        # Fallback: hash the shape and first few rows
        try:
            content_hash = hashlib.md5(str(df.head(50).values.tolist()).encode()).hexdigest()
            return f"{len(df)}_{content_hash}"
        except:
            return f"{len(df)}_{datetime.now().timestamp()}"

def compute_object_hash(obj: Any) -> Optional[str]:
    """Compute a hash for a generic object (dict, list, etc.)"""
    if obj is None:
        return None
    try:
        obj_str = json.dumps(obj, sort_keys=True, default=str)
        return hashlib.md5(obj_str.encode()).hexdigest()
    except:
        return str(hash(str(obj)))[:32]

def get_changed_keys(module_name: str) -> List[str]:
    """
    Get list of keys that have changed for a specific module
    Only saves records that have actually changed
    """
    if module_name not in MODULE_CONFIG:
        return []
    
    module_config = MODULE_CONFIG[module_name]
    tracking_keys = module_config.get("keys", [])
    
    hash_tracking = load_hash_tracking()
    changed_keys = []
    
    for key in tracking_keys:
        if key in st.session_state:
            current_value = st.session_state[key]
            
            # Compute current hash based on type
            if isinstance(current_value, pd.DataFrame):
                current_hash = compute_df_hash(current_value)
            elif isinstance(current_value, (dict, list)):
                current_hash = compute_object_hash(current_value)
            else:
                current_hash = str(current_value) if current_value else None
            
            previous_hash = hash_tracking.get(key)
            
            if current_hash != previous_hash:
                changed_keys.append(key)
                print(f"🔄 Changed detected: {key}")
    
    return changed_keys

def update_hash_tracking(module_name: str, saved_keys: List[str] = None):
    """Update hash tracking after saving"""
    if module_name not in MODULE_CONFIG:
        return
    
    module_config = MODULE_CONFIG[module_name]
    tracking_keys = module_config.get("keys", [])
    
    hash_tracking = load_hash_tracking()
    
    keys_to_update = saved_keys if saved_keys else tracking_keys
    
    for key in keys_to_update:
        if key in st.session_state:
            current_value = st.session_state[key]
            
            if isinstance(current_value, pd.DataFrame):
                hash_tracking[key] = compute_df_hash(current_value)
            elif isinstance(current_value, (dict, list)):
                hash_tracking[key] = compute_object_hash(current_value)
            elif current_value is not None:
                hash_tracking[key] = str(current_value)
            else:
                hash_tracking[key] = None
    
    save_hash_tracking(hash_tracking)

# ============================================================================
# SAVE FUNCTIONALITY - Save only changed records to pickle
# ============================================================================

def save_module_to_pickle(module_name: str, force_save_all: bool = False) -> Dict[str, Any]:
    """
    Save a specific module's data to pickle files.
    Only saves records that have changed (unless force_save_all=True).
    Returns statistics about what was saved.
    """
    if module_name not in MODULE_CONFIG:
        return {"error": f"Unknown module: {module_name}"}
    
    module_config = MODULE_CONFIG[module_name]
    save_keys = module_config.get("keys", [])
    save_files = module_config.get("pickle_files", [])
    
    # Get changed keys
    changed_keys = get_changed_keys(module_name) if not force_save_all else save_keys
    
    if not changed_keys and not force_save_all:
        return {
            "module": module_name,
            "saved_count": 0,
            "skipped_count": len(save_keys),
            "changed_keys": [],
            "message": "No changes detected"
        }
    
    saved_count = 0
    saved_files = []
    skipped_count = 0
    
    # Create a mapping from key to filename
    key_to_file = dict(zip(save_keys, save_files)) if len(save_keys) == len(save_files) else {}
    
    for key in save_keys:
        if key not in changed_keys and not force_save_all:
            skipped_count += 1
            continue
        
        if key in st.session_state:
            value = st.session_state[key]
            file_name = key_to_file.get(key, f"{key}.pkl")
            file_path = CACHE_DIR / file_name
            
            try:
                # Serialize and save
                with open(file_path, 'wb') as f:
                    pickle.dump(value, f)
                saved_count += 1
                saved_files.append(file_name)
                print(f"✅ Saved {key} to {file_name}")
            except Exception as e:
                print(f"❌ Error saving {key}: {e}")
    
    # Update hash tracking for saved keys
    update_hash_tracking(module_name, changed_keys if not force_save_all else save_keys)
    
    # Record save manifest
    record_save_manifest(module_name, saved_files, force_save_all)
    
    return {
        "module": module_name,
        "module_display": module_config["display_name"],
        "saved_count": saved_count,
        "skipped_count": skipped_count,
        "saved_files": saved_files,
        "changed_keys": changed_keys if not force_save_all else save_keys,
        "force_save": force_save_all,
        "timestamp": datetime.now().isoformat()
    }

def save_all_modules_to_pickle(force_save_all: bool = False) -> List[Dict[str, Any]]:
    """Save all modules that have data in session state"""
    results = []
    
    for module_name, module_config in MODULE_CONFIG.items():
        # Check if this module has any data in session state
        has_data = False
        for key in module_config.get("keys", []):
            if key in st.session_state:
                val = st.session_state[key]
                if isinstance(val, pd.DataFrame):
                    if not val.empty:
                        has_data = True
                        break
                elif val:
                    has_data = True
                    break
        
        if has_data or force_save_all:
            result = save_module_to_pickle(module_name, force_save_all)
            results.append(result)
    
    return results

# ============================================================================
# LOAD FUNCTIONALITY - Load from pickle files
# ============================================================================

def load_module_from_pickle(module_name: str, clear_existing: bool = True) -> Dict[str, Any]:
    """
    Load a specific module's data from pickle files.
    Optionally clears existing data before loading.
    """
    if module_name not in MODULE_CONFIG:
        return {"error": f"Unknown module: {module_name}"}
    
    module_config = MODULE_CONFIG[module_name]
    load_keys = module_config.get("keys", [])
    load_files = module_config.get("pickle_files", [])
    
    loaded_count = 0
    loaded_keys = []
    failed_files = []
    
    # Clear existing data if requested
    if clear_existing:
        for key in load_keys:
            if key in st.session_state:
                if isinstance(st.session_state[key], pd.DataFrame):
                    st.session_state[key] = pd.DataFrame()
                elif isinstance(st.session_state[key], dict):
                    st.session_state[key] = {}
                elif isinstance(st.session_state[key], list):
                    st.session_state[key] = []
                else:
                    st.session_state[key] = None
    
    # Create mapping from file to key
    file_to_key = dict(zip(load_files, load_keys)) if len(load_files) == len(load_keys) else {}
    
    for file_name in load_files:
        file_path = CACHE_DIR / file_name
        key = file_to_key.get(file_name)
        
        if not key:
            continue
        
        if file_path.exists():
            try:
                with open(file_path, 'rb') as f:
                    loaded_data = pickle.load(f)
                
                st.session_state[key] = loaded_data
                loaded_count += 1
                loaded_keys.append(key)
                print(f"✅ Loaded {key} from {file_name}")
            except Exception as e:
                failed_files.append(file_name)
                print(f"❌ Error loading {file_name}: {e}")
        else:
            print(f"⚠️ File not found: {file_name}")
    
    # Update hash tracking after load (to avoid false change detection)
    update_hash_tracking(module_name, loaded_keys)
    
    return {
        "module": module_name,
        "module_display": module_config["display_name"],
        "loaded_count": loaded_count,
        "loaded_keys": loaded_keys,
        "failed_files": failed_files,
        "timestamp": datetime.now().isoformat()
    }

def load_all_modules_from_pickle(clear_existing: bool = True) -> List[Dict[str, Any]]:
    """Load all modules that have pickle files"""
    results = []
    
    for module_name in MODULE_CONFIG.keys():
        result = load_module_from_pickle(module_name, clear_existing)
        if result["loaded_count"] > 0 or result["failed_files"]:
            results.append(result)
    
    return results

# ============================================================================
# MANIFEST MANAGEMENT - Track who saved what and when
# ============================================================================

def record_save_manifest(module_name: str, saved_files: List[str], force_save: bool = False):
    """Record save operation in manifest for audit trail"""
    manifest = load_manifest()
    
    user_name = "unknown"
    if 'user' in st.session_state:
        user_name = st.session_state['user'].get('username', 'unknown')
    
    entry = {
        "timestamp": datetime.now().isoformat(),
        "user": user_name,
        "module": module_name,
        "module_display": MODULE_CONFIG.get(module_name, {}).get("display_name", module_name),
        "saved_files": saved_files,
        "file_count": len(saved_files),
        "force_save": force_save,
        "session_id": st.session_state.get('session_id', str(uuid.uuid4()))
    }
    
    manifest["saves"].append(entry)
    manifest["last_updated"] = datetime.now().isoformat()
    
    # Keep only last 100 entries
    if len(manifest["saves"]) > 100:
        manifest["saves"] = manifest["saves"][-100:]
    
    save_manifest(manifest)

def load_manifest() -> Dict[str, Any]:
    """Load the save manifest"""
    if MANIFEST_FILE.exists():
        try:
            with open(MANIFEST_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    
    return {
        "saves": [],
        "last_updated": None,
        "total_saves": 0
    }

def save_manifest(manifest: Dict[str, Any]):
    """Save the manifest"""
    try:
        manifest["total_saves"] = len(manifest.get("saves", []))
        with open(MANIFEST_FILE, 'w') as f:
            json.dump(manifest, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving manifest: {e}")
        return False

def get_recent_saves(limit: int = 20) -> List[Dict[str, Any]]:
    """Get recent save operations from manifest"""
    manifest = load_manifest()
    saves = manifest.get("saves", [])
    # Return in reverse chronological order
    return list(reversed(saves[-limit:]))

# ============================================================================
# CACHE MANAGEMENT - View and clear cache
# ============================================================================

def get_cache_info() -> Dict[str, Any]:
    """Get information about all cache files"""
    total_size = 0
    files_info = []
    
    # Check pickle files
    for file_path in CACHE_DIR.glob("*.pkl"):
        size = file_path.stat().st_size
        total_size += size
        files_info.append({
            "name": file_path.name,
            "size_bytes": size,
            "size_mb": size / (1024 * 1024),
            "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
            "type": "pickle"
        })
    
    # Check hash tracking file
    if HASH_TRACKING_FILE.exists():
        size = HASH_TRACKING_FILE.stat().st_size
        total_size += size
        files_info.append({
            "name": HASH_TRACKING_FILE.name,
            "size_bytes": size,
            "size_mb": size / (1024 * 1024),
            "modified": datetime.fromtimestamp(HASH_TRACKING_FILE.stat().st_mtime).isoformat(),
            "type": "hash_tracking"
        })
    
    return {
        "total_files": len(files_info),
        "total_size_bytes": total_size,
        "total_size_mb": total_size / (1024 * 1024),
        "files": files_info,
        "cache_dir": str(CACHE_DIR)
    }

def clear_all_cache():
    """Clear all cache files"""
    cleared_files = []
    
    for file_path in CACHE_DIR.glob("*.pkl"):
        try:
            os.remove(file_path)
            cleared_files.append(file_path.name)
        except:
            pass
    
    if HASH_TRACKING_FILE.exists():
        try:
            os.remove(HASH_TRACKING_FILE)
            cleared_files.append(HASH_TRACKING_FILE.name)
        except:
            pass
    
    # Reset hash tracking
    save_hash_tracking({})
    
    return cleared_files

def clear_module_cache(module_name: str) -> List[str]:
    """Clear cache for a specific module"""
    if module_name not in MODULE_CONFIG:
        return []
    
    module_config = MODULE_CONFIG[module_name]
    cleared_files = []
    
    for file_name in module_config.get("pickle_files", []):
        file_path = CACHE_DIR / file_name
        if file_path.exists():
            try:
                os.remove(file_path)
                cleared_files.append(file_name)
            except:
                pass
    
    return cleared_files

# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_cache_management_ui():
    """Render the cache management UI in sidebar"""
    st.markdown("---")
    st.markdown("### 💾 Cache Management")
    
    # Cache info
    cache_info = get_cache_info()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📁 Cache Files", cache_info["total_files"])
    with col2:
        st.metric("💾 Cache Size", f"{cache_info['total_size_mb']:.2f} MB")
    
    # Clear cache button
    if st.button("🗑️ Clear All Cache", use_container_width=True):
        cleared = clear_all_cache()
        if cleared:
            st.success(f"✅ Cleared {len(cleared)} cache files")
            st.rerun()
        else:
            st.info("No cache files to clear")
    
    # Show cache details in expander
    with st.expander("📋 Cache Details"):
        if cache_info["files"]:
            for file_info in cache_info["files"]:
                st.caption(f"📄 {file_info['name']} - {file_info['size_mb']:.2f} MB")
        else:
            st.caption("No cache files found")

def render_save_load_ui(module_name: str = None):
    """
    Render save/load buttons for a specific module or all modules
    """
    st.markdown("### 💾 Manual Save/Load")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save Changes Only", use_container_width=True, help="Save only records that have changed"):
            if module_name:
                result = save_module_to_pickle(module_name, force_save_all=False)
                if result.get("saved_count", 0) > 0:
                    st.success(f"✅ Saved {result['saved_count']} changed datasets")
                else:
                    st.info(result.get("message", "No changes detected"))
            else:
                results = save_all_modules_to_pickle(force_save_all=False)
                total_saved = sum(r.get("saved_count", 0) for r in results)
                if total_saved > 0:
                    st.success(f"✅ Saved {total_saved} changed datasets across {len(results)} modules")
                else:
                    st.info("No changes detected in any module")
    
    with col2:
        if st.button("💾 Save All (Force)", use_container_width=True, help="Save all records regardless of changes"):
            if module_name:
                result = save_module_to_pickle(module_name, force_save_all=True)
                st.success(f"✅ Saved {result['saved_count']} datasets (forced)")
            else:
                results = save_all_modules_to_pickle(force_save_all=True)
                total_saved = sum(r.get("saved_count", 0) for r in results)
                st.success(f"✅ Saved {total_saved} datasets across {len(results)} modules (forced)")
    
    with col3:
        if st.button("📂 Load from Cache", use_container_width=True, help="Load previously saved data from cache"):
            if module_name:
                result = load_module_from_pickle(module_name)
                if result["loaded_count"] > 0:
                    st.success(f"✅ Loaded {result['loaded_count']} datasets")
                    st.rerun()
                else:
                    st.warning("No data found in cache")
            else:
                results = load_all_modules_from_pickle()
                total_loaded = sum(r.get("loaded_count", 0) for r in results)
                if total_loaded > 0:
                    st.success(f"✅ Loaded {total_loaded} datasets across {len(results)} modules")
                    st.rerun()
                else:
                    st.warning("No data found in cache")

def render_recent_saves_ui():
    """Render recent save operations in sidebar"""
    st.markdown("### 📋 Recent Saves")
    
    recent_saves = get_recent_saves(5)
    
    if recent_saves:
        for save in recent_saves:
            timestamp = save.get("timestamp", "")[:16].replace("T", " ")
            user = save.get("user", "unknown")
            module = save.get("module_display", save.get("module", "unknown"))
            files = save.get("file_count", 0)
            
            st.caption(f"📌 {timestamp}")
            st.caption(f"   👤 {user} | 📁 {module} | 📄 {files} files")
            st.caption("---")
    else:
        st.caption("No saves recorded yet")

def render_manual_controls_sidebar():
    """Render all manual controls in sidebar"""
    st.markdown("---")
    st.markdown("### 🎮 Manual Controls")
    
    # Module selector for targeted operations
    module_options = ["All Modules"] + [MODULE_CONFIG[m]["display_name"] for m in MODULE_CONFIG.keys()]
    selected_module_display = st.selectbox(
        "Select Module",
        options=module_options,
        help="Choose which module to save/load, or 'All Modules' for everything"
    )
    
    # Convert display name back to module key
    if selected_module_display == "All Modules":
        selected_module = None
    else:
        for key, config in MODULE_CONFIG.items():
            if config["display_name"] == selected_module_display:
                selected_module = key
                break
        else:
            selected_module = None
    
    # Save/Load buttons
    render_save_load_ui(selected_module)
    
    # Change tracking indicator
    st.markdown("---")
    st.markdown("### 🔄 Change Tracking")
    
    # Show which modules have changes
    hash_tracking = load_hash_tracking()
    for module_name, module_config in MODULE_CONFIG.items():
        changed_keys = get_changed_keys(module_name)
        if changed_keys:
            st.warning(f"🟡 {module_config['display_name']}: {len(changed_keys)} change(s)")
        else:
            st.success(f"🟢 {module_config['display_name']}: Synced")
    
    # Force save all button
    st.markdown("---")
    if st.button("⚡ Force Save ALL Modules", use_container_width=True, type="primary"):
        with st.spinner("Saving all modules..."):
            results = save_all_modules_to_pickle(force_save_all=True)
            total_saved = sum(r.get("saved_count", 0) for r in results)
            st.success(f"✅ Saved {total_saved} datasets across {len(results)} modules")
    
    # Recent saves
    render_recent_saves_ui()
    
    # Cache management
    render_cache_management_ui()
    
    # Reset change tracking button
    st.markdown("---")
    if st.button("🔄 Reset Change Tracking", use_container_width=True, help="Reset all hash tracking. Next save will save all records."):
        save_hash_tracking({})
        st.success("Change tracking reset! Next save will save all records.")