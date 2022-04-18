import Home from './pages/Home';
import './App.css';
import React, { useState, useEffect } from "react";
import axios from "axios";

// import {BrowerRouter as Router, Route, Switch} from 'react-router-dom';

function App() {

  const [formData, setFormData] = useState(null)

  function getData() {
    axios({
      method: "POST",
      url:"/profile",
    })
    .then((response) => {
      const res =response.data
      setProfileData(({
        profile_name: res.name,
        about_me: res.about}))
    }).catch((error) => {
      if (error.response) {
        console.log(error.response)
        console.log(error.response.status)
        console.log(error.response.headers)
        }
    })}

    return (
      <div className="App">
        <header className="App-header">
          <img src={logo} className="App-logo" alt="logo" />
          <p>
            Edit <code>src/App.js</code> and save to reload.
          </p>
          <a
            className="App-link"
            href="https://reactjs.org"
            target="_blank"
            rel="noopener noreferrer"
          >
            Learn React
          </a>
  
          {/* new line start*/}
          <p>To get your profile details: </p><button onClick={getData}>Click me</button>
          {profileData && <div>
                <p>Profile name: {profileData.profile_name}</p>
                <p>About me: {profileData.about_me}</p>
              </div>
          }
           {/* end of new line */}
        </header>
      </div>
    );

  // return (
  //   <div>
  //     <Home />
  //   </div>
  // );
}

export default App;
