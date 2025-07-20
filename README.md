Step 1: Go to Start click on the mySQL workbench to run
Step 2: select the root via localhost 3309 
Step 3: You will be prompted to enter the password {}
Step 4: right click on the tables in the accounts database to drop a table or import a new one
Step 5: while importing make sure the date formats align to the date formats in the code i.e no words e.g jul in the bank tables
Step 6: also ensure the top margin of every excel file is the heading of a column
Step 7: Adjust the date of the fx_trade_tracker in the code in line 6 of the python script 
Step 8; Adjust the date of the fx_date str to match the date of the bank tables line 10 of the python script 
Step 9: Next press the RUN button in the vs code to match the accounts details/print via cmd by navigating to the same directory and running the command {python3 verify_transfers.py}
Step 10: Navigate to {cd C:/Desktop/verify_transfers} to view the printed results 

NB:
files must be in csv format
Top rows must have headers
the bank tables must be named according the format in the code 



debit=debit/withdrawal
credit=credit/deposit
naah that didnt work anyways lemme rename the column to be counterparty /choice payment so in our code myunknowncolumn will be counterparty and myunknown column 2 