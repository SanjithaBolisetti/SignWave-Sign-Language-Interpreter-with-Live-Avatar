import React from "react";

function Intro() {
  return (
    <section id="intro">
      <div className="container">
        <div className="row my-5">
          <div
            className="col-md-12 d-flex justify-content-center align-items-center"
            style={{ flexDirection: "column" }}
          >
            <div className="h2 section-heading">
              Welcome to <span style={{ color: "#FACC15" }}>SignWave</span>
            </div>

            <div className="col-lg-4 divider my-3" />

            <div className="text-center normal-text">
              <strong>SignWave</strong> is an intelligent and inclusive platform
              designed to bridge communication gaps using
              <strong> Indian Sign Language (ISL)</strong>.
              <br /><br />
              Convert text and speech into expressive sign animations, explore
              ISL alphabets and commonly used words, and interact with lifelike
              3D avatars — all through a clean, distraction-free interface.
              <br /><br />
              Built with accessibility, simplicity, and impact in mind,
              SignWave empowers both learners and users to communicate beyond
              barriers.
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

export default Intro;
