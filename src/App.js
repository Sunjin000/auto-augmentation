import Home from './pages/Home';
import './App.css';
import React, { useState, useEffect } from "react";
import axios from "axios";
import logo from './logo.svg';

// import {BrowerRouter as Router, Route, Switch} from 'react-router-dom';

function App() {
  useEffect(() => {
    fetch('/home', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ })
    })
      .then((response) => response.text())
      .then((data) => console.log(data));
  }, []);

  return (
    <div>
      <Home />
    </div>
  );
}


export default App;
