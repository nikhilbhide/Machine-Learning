## happy path
* greet
  - utter_greet
* mood_great
  - utter_happy

## sad path 1
* greet
  - utter_greet
* mood_unhappy
  - utter_cheer_up
  - utter_did_that_help
* affirm
  - utter_happy

## sad path 2
* greet
  - utter_greet
* mood_unhappy
  - utter_cheer_up
  - utter_did_that_help
* deny
  - utter_goodbye

## say goodbye
* goodbye
  - utter_goodbye

## bot challenge
* bot_challenge
  - utter_iamabot

## interactive_story_1
* greet
    - utter_ask_for_order
* utter_ask_location
    - utter_ask_for_location
* set_location
    - set_location
    - slot{"location": "pune"}
    - utter_ask_cuisine
* cuisine_identifier
    - action_set_cuisine
    - slot{"cuisine": "Mexican"}
    - utter_ask_for_range
* deny
    - action_set_price_range
    - slot{"minPrice": 0}
    - slot{"maxPrice": 299}
    - utter_goodbye

## interactive_story_1
* greet
    - utter_ask_for_order
* utter_ask_location
    - utter_ask_for_location
* location_identifier
    - set_location
    - slot{"location": "mumbai"}
    - utter_ask_cuisine
* mood_great
    - action_set_price_range
    - slot{"minPrice": 0}
    - slot{"maxPrice": -999}
    - utter_goodbye

## interactive_story_1
* greet
    - utter_ask_for_order
* utter_ask_location
    - utter_ask_for_location
* location_identifier
    - set_location
    - slot{"location": "mumbai"}
    - utter_ask_cuisine
* bot_challenge
    - action_set_cuisine
    - slot{"cuisine": "Italian"}
    - utter_ask_for_range
* deny
    - action_set_price_range
    - slot{"minPrice": 300}
    - slot{"maxPrice": 700}
    - action_suggest_restaurants
    - utter_goodbye

## interactive_story_1
* greet
    - utter_ask_for_order
* utter_ask_location
    - utter_ask_for_location
* bot_challenge
    - set_location
    - slot{"location": "kolkata"}
    - utter_ask_cuisine
* bot_challenge
    - action_set_cuisine
    - slot{"cuisine": "South Indian"}
    - utter_ask_for_range
* deny
    - action_set_price_range
    - slot{"minPrice": 0}
    - slot{"maxPrice": -999}
    - action_suggest_restaurants
    - utter_goodbye

## interactive_story_1
* greet
    - utter_ask_for_order
* utter_ask_location
    - utter_ask_for_location
* set_location
    - set_location
    - slot{"location": "pune"}
    - utter_ask_cuisine
* cuisine_identifier
    - action_set_cuisine
    - slot{"cuisine": "Mexican"}
    - utter_ask_for_range
* deny
    - action_set_price_range
    - slot{"minPrice": 300}
    - slot{"maxPrice": 700}
    - utter_ask_mail
* mail
    - set_receiver_mail_address
    - slot{"receiverMailAddress": "upgradnikhil@gmail.com"}
    - action_suggest_restaurants
    - utter_goodbye

## interactive_story_1
* greet
    - utter_ask_for_order
* utter_ask_location
    - utter_ask_for_location
* set_location
    - set_location
    - slot{"location": "pune"}
    - utter_ask_cuisine
* cuisine_identifier
    - action_set_cuisine
    - slot{"cuisine": "Mexican"}
    - utter_ask_for_range
* deny
    - action_set_price_range
    - slot{"minPrice": 300}
    - slot{"maxPrice": 700}
    - utter_ask_mail
* mail
    - set_receiver_mail_address
    - slot{"receiverMailAddress": "upgradnikhil@gmail.com"}
    - action_suggest_restaurants
    - utter_goodbye

## interactive_story_1
* greet
    - utter_ask_for_order
* utter_ask_location
    - utter_ask_for_location
* greet
    - set_location
    - slot{"location": "delhi"}
    - utter_ask_cuisine
* cuisine_identifier
    - action_set_cuisine
    - slot{"cuisine": "Mexican"}
    - utter_ask_for_range
* deny
    - action_set_price_range
    - slot{"minPrice": 300}
    - slot{"maxPrice": 700}
    - utter_ask_mail
* mail
    - set_receiver_mail_address
    - slot{"receiverMailAddress": "upgradnikhil@gmail.com"}
    - action_suggest_restaurants
    - utter_goodbye
