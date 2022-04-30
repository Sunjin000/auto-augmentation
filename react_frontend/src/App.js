import Home from './pages/Home'
import Confirm from './pages/Confirm'
import Progress from './pages/Progress'
import Result from './pages/Result'
import Error from './pages/Error'
import './App.css';
import React, { useState, useEffect } from "react";
import axios from "axios";
import {BrowserRouter, Route, Routes, Switch} from 'react-router-dom';

function App() {
  return (
    <div>
      <BrowserRouter>
        <Routes>
          <Route exact path="/" element={<Home/>}/>
          <Route exact path="/confirm" element={<Confirm/>}/>
          <Route exact path="/progress" element={<Progress/>}/>
          <Route exact path="/result" element={<Result/>}/>
          <Route exact path="/error" element={<Error/>}/>
        </Routes>
      </BrowserRouter>
    </div>
  );

}


export default App;
