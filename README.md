# 📍 Interactive Relationship Mapper

This Streamlit app helps you map and visualize your relationships using a network graph. You can either **manually input people and their connections** or **upload a JSON file** that follows a specific format.
You can try it yourself at the following link: **https://relationshipmapping-8qpjg4kqwudamemnmwodub.streamlit.app/**

## 🌟 Features

- Visualize your network of friends or contacts
- Add people manually or upload a JSON file
- Automatically draw connections between people
- Group and color nodes by shared attributes (e.g., where you met, hobbies)
- Export your graph as an interactive HTML or PNG file
- Clean and simple UI powered by Streamlit and Plotly

## 🧠 JSON Format Example

If you choose to upload a JSON file, make sure it follows this format:

```json
{
  "people": [
    {
      "first_name": "Pou",
      "last_name": "Let",
      "age": 32,
      "sex": "M",
      "hobby": "Volleyball",
      "job": "QA Engineer",
      "met_at": "HS",
      "connections": ["Alice A", "Anais V"]
    },
    {
      "first_name": "Alice",
      "last_name": "A",
      "age": 30,
      "sex": "F",
      "hobby": "Travel",
      "job": "Consultant",
      "met_at": "Work",
      "connections": ["Pou Let"]
    }
  ]
}
