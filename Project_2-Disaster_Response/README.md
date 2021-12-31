Udacity PROJECT: Disaster Response.

In this project our perpose is to take messages and analyse them to detect fast and efficiently what is real, what is needed and how to respond to these messages.

Data is provided by Figure eight and contains messages from texts or tweets from real life disasters. The script is cleaning the data, conneccting the categories and eventually running a machine learning pipeline to identify to which messages to respond. A final report is produced using the model created and classifies a message the user puts directly to the relevant categories for fast response.

example message and response:
![image](https://user-images.githubusercontent.com/75398467/147814654-2543f074-cbf1-40cb-9d25-b8f598d19c08.png)



### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python app/run.py`

3. Go to http://0.0.0.0:3001/


### Requirements for the script and report
All needed modules are automatically installed to run for the script. 

In case the report is not loading please follow below instructions provided by the Udacity discussion forum:
 - Once your app is running (python run.py)
 - Open another terminal and type env|grep WORK this will give you the spaceid it will start with view*** and some characters after that
 - Now open your browser window and type https://viewa7a4999b-3001.udacity-student-workspaces.com, replace the whole viewa7a4999b with your space id you got in the step 2
 - Press enter and the app should now run for you
 
Local Machine
 - Once your app is running (python run.py)
 - Go to to localhost:3001 and app will run
