import Home from './pages/Home';
import './App.css';
import React, { useState, useEffect } from "react";
import axios from "axios";
import logo from './logo.svg';

// import {BrowerRouter as Router, Route, Switch} from 'react-router-dom';

function App() {
  // new line start
 const [profileData, setProfileData] = useState(null)

 function getData() {
   axios({
     method: "GET",
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
   //end of new line 

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
}
//   useEffect(() => {
//     fetch('/', {
//       method: 'POST',
//       headers: { 'Content-Type': 'application/json' },
//       body: JSON.stringify({ })
//     })
//       .then((response) => response.json())
//       .then((data) => console.log(data));
//   }, []);

//   return (
//     <div>
//       <Home />
//     </div>
//   );
// }


export default App;
