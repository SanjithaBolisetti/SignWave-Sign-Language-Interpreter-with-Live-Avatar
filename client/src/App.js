import "./App.css";
import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

import Home from "./Pages/Home";
import Convert from "./Pages/Convert";
import Navbar from "./Components/Navbar";
import Footer from "./Components/Footer";
import SignToText from "./Pages/SignToText";


function App() {
  return (
    <Router>
      <Navbar />
      <Routes>
        <Route path="/signwave/home" element={<Home />} />
        <Route path="/signwave/explore" element={<Convert />} />
        <Route path="/signwave/sign-to-text" element={<SignToText />} />


        <Route path="*" element={<Home />} />
      </Routes>
      <Footer />
    </Router>
  );
}

export default App;
