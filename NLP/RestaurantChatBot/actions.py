# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/core/actions/#custom-actions/


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List
#
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
import zomatopy
import pprint, json
import requests
from pprint import pprint


#
#
class RestaurantAPI(object):
    def search(self, output):
        results = []
        counter = 0

        for restaurants in output["restaurants"]:
            for key in restaurants:
                restaurantName = restaurants[key]["name"]
                location = restaurants[key]["location"]
                locality = location["locality"]
                print(locality)
                result =  "Restaunrat %s in locality %s"%(restaurantName,locality)
                results.append(result)
                
                #break the loop after retreiving 10 results
                if counter==10:
                    break
        return results

    def sendMail(self, mailMessage, receiverMail):
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        mail_content = "Hello"

        #The mail addresses and password
        sender_address = 'upgradnikhil@gmail.com'
        sender_pass = 'izmahyrddrnwuenw'
        receiver_address = receiverMail
        #Setup the MIME 
        message = MIMEMultipart()
        message['From'] = sender_address
        message['To'] = receiver_address
        message['Subject'] = 'A test mail sent by Python. It has an attachment.'   #The subject line
        #The body and the attachments for the mail
        message.attach(MIMEText(mail_content, 'plain'))
        #Create SMTP session for sending the mail
        session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
        session.starttls() #enable security
        session.login(sender_address, sender_pass) #login with mail_id and password
        session.sendmail(sender_address, receiver_address, mailMessage)
        session.quit()
        print('Mail Sent')

class MailLocator(Action):
    def name(self) -> Text:
            return "set_receiver_mail_address"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        receiverMail = tracker.latest_message.get('text')
        print(receiverMail)
        return[SlotSet("receiverMailAddress",  receiverMail)]
    
class Locator(Action):

    def name(self) -> Text:
        return "set_location"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message("Hello World!")
        location = tracker.latest_message.get('text')
        print(location)
        return[SlotSet("location",  location)]

class Cuisine(Action):
    
    def name(self) -> Text:
        return "action_set_cuisine"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message("Hello World!")
        cuisine = tracker.latest_message.get('text')
        print(cuisine)
        return[SlotSet("cuisine",  cuisine)]

class PriceRange(Action):
    
    def name(self) -> Text:
        return "action_set_price_range"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message("Hello World!")
        priceRange = tracker.latest_message.get('text')
        minPrice = 0
        maxPrice = 0
        print(priceRange)
        if priceRange =="Lesser than Rs. 300":
            minPrice = 0
            maxPrice = 299
        elif priceRange == "Rs. 300 to 700":
            minPrice = 300
            maxPrice = 700
        else:
            minPrice=minPrice
            maxPrice=-999
        return[SlotSet("minPrice",  minPrice),SlotSet("maxPrice",  maxPrice)]

class RestaurantLocator(Action):
    def name(self) -> Text:
        return "action_suggest_restaurants"
  
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        config={
            "user_key":"e495869da95df8533751c2bbb5c95fe7"
        }

        #initialize the zomatopy instance
        zomato = zomatopy.initialize_app(config)
        
        #retreive the city id and cuisine ids from zomato for corresponding cuisine and location
        location = tracker.get_slot("location")
        cuisine = tracker.get_slot("cuisine")
        receiverMailAddress = tracker.get_slot("receiverMailAddress")

        cityID = zomato.get_city_ID(location)
        
        #retrieve the cuisine and get the id
        cuisineDict = zomato.get_cuisines(cityID)
        cuisineID = 0
        for id, value in cuisineDict.items(): 
            if (value==cuisine):
                cuisineID = id
  
        print(cuisineID)

        #search the restaurants for cityid and cuisine
        zomatoURL = "https://developers.zomato.com/api/v2.1/search?entity_id=%d&entity_type=city&cuisines=%d"%(cityID,cuisineID)
        header = {"User-agent": "curl/7.43.0", "Accept": "application/json", "user_key": "e495869da95df8533751c2bbb5c95fe7"}
        response = requests.get(zomatoURL, headers=header)
        binary = response.content
        output = json.loads(binary)
        restaurant_api = RestaurantAPI()
        results = restaurant_api.search(output)
        sep = "\n"
        outputText = sep.join(results).encode('utf-8')
        dispatcher.utter_message(sep.join(results))
        restaurant_api.sendMail(outputText,receiverMailAddress)
        print(output["results_found"])
