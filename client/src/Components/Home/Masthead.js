import React from "react";

function Masthead() {
  return (
    <div className="container-fluid hero-section">
      <div className="container text-center">
        <h1 className="display-1 fw-bold">
          Sign<span style={{ color: "#FF7A59" }}>Wave</span>
        </h1>
        <p className="lead mt-3">
          Transform text and speech into expressive Indian Sign Language
          using intelligent 3D avatars.
        </p>
        <a href="#services" className="btn btn-primary btn-lg mt-4">
          Explore SignWave
        </a>
      </div>
    </div>
  );
}

export default Masthead;
