## Running backend

  Install docker
  Clone the repository
  Navigate to the Directory
  Enter "docker-compose up"

## Notes

  I have left some comments in the code, indicating potential areas for improvement or highlighting attempted fixes for encountered issues. Feel free to remove them at your discretion.

## Functionality

  Data has been processed before being worked upon as it is always a good idea to make sure the data we have is clean, relevant, and ready to be worked upon. Cleaner data leads to better performance and functionality.
  Added funtionality for creating summary.

## Summary Creation Functionality:

  The summarization model used is 't5-small.' It is suitable for smaller models and limited computation power.
  For more powerful hardware, consider alternative models such as 't5-large,' 'BART,' or 'Pegasus' for the summarizer. Adjust the max_length parameter in the summarize_text function accordingly.
  Be cautious with resource-intensive models, as it may impact Docker's performance. Adjust Docker settings or choose a less resource-intensive model if needed.


## Further possible work

  Given more time, One could extend the functionality by implementing additional features like question answering, key topic identification, etc.

 


