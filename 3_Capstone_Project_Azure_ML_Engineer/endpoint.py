import requests
import json

# URL for the web service, should be similar to:
# 'http://8530a665-66f3-49c8-a953-b82a2d312917.eastus.azurecontainer.io/score'
scoring_uri = 'http://b1823125-1cf8-447f-9ea6-bd306d25e911.westeurope.azurecontainer.io/score'
# If the service is authenticated, set the key or token
#key = '5pz3UEsZ2XXNG3Wt6gUiiUtgTNWjpG7E'

# Two sets of data to score, so we get two results back
'''
data = {"data": 
    [
      {
        'age':75,
        'anaemia':0,
        'creatinine_phosphokinase':582,
        'diabetes':0,
        'ejection_fraction':20, 
        'high_blood_pressure':1, 
        'platelets':265000,
        'serum_creatinine':1.9, 
        'serum_sodium':130, 
        'sex':1, 
        'smoking':0, 
        'time':4
        }, 
    ]
  }
'''
data = {
  "Inputs":{
      "data": 
    [
      {
        'age':75,
        'anaemia':0,
        'creatinine_phosphokinase':582,
        'diabetes':0,
        'ejection_fraction':20, 
        'high_blood_pressure':1, 
        'platelets':265000,
        'serum_creatinine':1.9, 
        'serum_sodium':130, 
        'sex':1, 
        'smoking':0, 
        'time':4
        }, 
    ],
  },
"GlobalParameters": {
      'method': "predict",
    }
  }
# Convert to JSON string
input_data = json.dumps(data)

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
#headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)

'''
{
    "data":[
        {
        "age":75,
        "anaemia":0,
        "creatinine_phosphokinase":582,
        "diabetes":0,
        "ejection_fraction":20, 
        "high_blood_pressure":1, 
        "platelets":265000,
        "serum_creatinine":1.9, 
        "serum_sodium":130, 
        "sex":1, 
        "smoking":0, 
        "time":4
        }
    ]
  }
'''