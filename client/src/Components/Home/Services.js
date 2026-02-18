import React from "react";
import { Link } from "react-router-dom";
import exploreImg from "../../Assets/explore-sign.png";

function Services() {
  return (
    <section id="services">
      <div className="container">
        <div className="row mt-5">
          <div className="col-md-12 text-center">
            <h2 className="section-heading">What You Can Do</h2>
            <div className="divider my-3 mx-auto" />
            <p className="normal-text">
              Explore Indian Sign Language through intelligent 3D animations
              designed for inclusive and accessible communication.
            </p>
          </div>
        </div>

        <div className="row justify-content-center">
          {/* Explore Now */}
          <div className="col-lg-5 mt-4">
            <div className="card h-100 text-center">
              <img
                src={exploreImg}
                className="card-img-top"
                alt="Explore Indian Sign Language"
              />
              <div className="card-body">
                <h5 className="card-title">Explore Now</h5>
                <p className="card-text">
                  Convert text or speech into expressive Indian Sign Language
                  using lifelike 3D avatars. Simple, fast, and accessible for
                  everyone.
                </p>
              </div>
              <div className="card-footer border-0">
                <Link to="/signwave/explore" className="btn btn-primary w-100">
                  Explore Now
                </Link>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

export default Services;
