############ Cources Web Srapping 
import bs4
import requests 
import re

url = "https://www.coursetalk.com/providers/udemy/courses"

page = requests.get(url)
soup = bs4.BeautifulSoup(page.text,'html.parser')

result = soup.findAll("div",{"class":"course-listing-card"})
#print(result[0])
course_name = []
course_description = []
course_review_nos = []
course_url = []
for res in result:
    c_name = res.find("span").text.strip()
    course_name.append(c_name)
    c_description = res.find("div",{"class":"course-listing__summary__description"}).text.strip()
    course_description.append(c_description)
    c_reviews = res.find("li",{"class":"course-listing-summary__ratings__number"}).text.strip()
    c_reviews = int(re.findall('\d*',str(c_reviews))[0])
    course_review_nos.append(c_reviews)
    c_url = res.find("a",{"class":"btn btn-md btn-success js-course-search-result"})
    c_url = "https://www.coursetalk.com" + str(re.findall('href="(.*)">',str(c_url))[0])
    course_url.append(c_url)
    #print(len(course_url))

for page_no in range(2,884):
    page_url = "https://www.coursetalk.com/providers/udemy/courses?filters=platform:udemy&page="+str(page_no)+"&sort=-ct_score"
    page = requests.get(page_url)
    soup = bs4.BeautifulSoup(page.text,'html.parser')
    result = soup.findAll("div",{"class":"course-listing-card"})
    for res in result:
        c_name = res.find("span").text.strip()
        course_name.append(c_name)
        c_description = res.find("div",{"class":"course-listing__summary__description"}).text.strip()
        course_description.append(c_description)
        c_reviews = res.find("li",{"class":"course-listing-summary__ratings__number"}).text.strip()
        c_reviews = int(re.findall('\d*',str(c_reviews))[0])
        course_review_nos.append(c_reviews)
        c_url = res.find("a",{"class":"btn btn-md btn-success js-course-search-result"})
        c_url = "https://www.coursetalk.com" + str(re.findall('href="(.*)">',str(c_url))[0])
        course_url.append(c_url)
        #print(len(course_url))
#print(len(course_name))
#print(course_name)
#print(len(course_description))
#print(course_description)
#print(len(course_review_nos))
#print(course_review_nos)
#print(len(course_url))
#print(course_url)