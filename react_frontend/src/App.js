import Home from './pages/Home'
import Confirm from './pages/Confirm'
import Progress from './pages/Progress'
import Result from './pages/Result'
import './App.css';
import React, { useState, useEffect } from "react";
import axios from "axios";
import {BrowserRouter, Route, Routes, Switch} from 'react-router-dom';

function App() {
  // useEffect(() => {
  //   console.log('print here')
  //   fetch('/home', {
  //     method: 'POST',
  //     headers: { 'Content-Type': 'application/json' },
  //     body: JSON.stringify({ })
  //   })
  //     .then((response) => response.text())
  //     .then((data) => console.log('data', data));
  // }, []);

  // useEffect(() => {
  //   fetch('/api').then(response =>{
  //     if(response.ok){
  //       return response.json()
  //     }
  //   }).then(data => console.log('api', data))
  // }, [])

  return (
    <div>
      {/* <Home /> */}
      {/* <Confirm /> */}
      {/* <Progress /> */}
      {/* <Result /> */}
      <BrowserRouter>
        <Routes>
          <Route exact path="/" element={<Home/>}/>
          <Route exact path="/confirm" element={<Confirm/>}/>
          <Route exact path="/progress" element={<Progress/>}/>
          <Route exact path="/result" element={<Result/>}/>
        </Routes>
      </BrowserRouter>
    </div>
  );

}


export default App;
