import requests

url = "https://stock-and-options-trading-data-provider.p.rapidapi.com/options/aapl"

headers = {
	"X-RapidAPI-Proxy-Secret": "a755b180-f5a9-11e9-9f69-7bf51e845926",
	"X-RapidAPI-Key": "e71e5cdf40msh721c5852fab69abp1d3c07jsn65f8c94f025f",
	"X-RapidAPI-Host": "stock-and-options-trading-data-provider.p.rapidapi.com"
}

response = requests.request("GET", url, headers=headers)

print(response.text)