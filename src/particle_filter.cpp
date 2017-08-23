/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using std::normal_distribution;
using std::default_random_engine;
using vector_t = std::vector<double>;
using namespace std;

default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  num_particles = 200;
  weights.resize(num_particles, 1.0);

  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for(int i = 0; i < num_particles; i++) {
    Particle par;
    par.id = i;
    par.x = dist_x(gen);
    par.y = dist_y(gen);
    par.theta = dist_theta(gen);
    
    par.weight = 1.0;
    particles.push_back(par);
  }

  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  normal_distribution<double> N_x(0, std_pos[0]);
  normal_distribution<double> N_y(0, std_pos[1]);
  normal_distribution<double> N_theta(0, std_pos[2]);
    
  for (int i = 0; i < num_particles; i++) {

    if (fabs(yaw_rate) < 0.00001){  
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    } 

    else {
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }
    
    // noise
    particles[i].x += N_x(gen);
    particles[i].y += N_y(gen);
    particles[i].theta += N_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

int min_i;
double min_distance, distance, dx, dy;

  for(int num = 0; num < observations.size(); num++) {
    const auto observation = observations[num];
    min_distance = 200;
    min_i = -1;

    for(int i = 0; i < predicted.size(); i++){
      auto pred_lm = predicted[i];
      dx = (pred_lm.x - observation.x);
      dy = (pred_lm.y - observation.y);
      distance = dx*dx + dy*dy;

      if(distance < min_distance){
        min_distance = distance;
        min_i = i;
      }
    }
    // index of landmark
    observations[num].id = min_i;
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

// landmark with uncertainty
double sigma_landmark [2] = {0.3, 0.3};

  for(int p_ctr=0; p_ctr < particles.size(); p_ctr++) {
    auto p = particles[p_ctr];

    std::vector<LandmarkObs> predicted_landmarks;

    for(auto lm : map_landmarks.landmark_list) {
      LandmarkObs lm_pred;
      lm_pred.x = lm.x_f;
      lm_pred.y = lm.y_f;
      lm_pred.id = lm.id_i;
      auto dx = lm_pred.x - p.x;
      auto dy = lm_pred.y - p.y;

      // check if in range
      if(dx*dx + dy*dy <= sensor_range*sensor_range)
        predicted_landmarks.push_back(lm_pred);
    }
    std::vector<LandmarkObs> transformed_obs;
    double total_prob = 1.0f;

    // transform coordinates
    for(auto obs_lm : observations) {
  
      LandmarkObs obs_all;
      obs_all.x = p.x + obs_lm.x * cos(p.theta) - obs_lm.y * sin(p.theta);
      obs_all.y = p.y + obs_lm.x * sin(p.theta) + obs_lm.y * cos(p.theta);
      obs_all.id = obs_lm.id;
      transformed_obs.push_back(std::move(obs_all));
    }

    dataAssociation(predicted_landmarks, transformed_obs);

    for(int i=0; i < transformed_obs.size(); i++) {
      auto observation = transformed_obs[i];
      auto assoc_lm = predicted_landmarks[observation.id];
      auto cov_x = sigma_landmark[0]*sigma_landmark[0];
      auto cov_y = sigma_landmark[1]*sigma_landmark[1];
      auto normalize = 2.0*M_PI*sigma_landmark[0]*sigma_landmark[1];
      auto dx = (observation.x - assoc_lm.x);
      auto dy = (observation.y - assoc_lm.y);
      auto norm_obs = exp(-(dx*dx/(2*cov_x) + dy*dy/(2*cov_y)))/normalize;
      total_prob *= norm_obs;
    }
    particles[p_ctr].weight = total_prob;
    weights[p_ctr] = total_prob;
  }
std::cout<<std::endl;

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  //  normalization of weights to get probability
  std::discrete_distribution<int> d(weights.begin(), weights.end());
  std::vector<Particle> new_particles;

  for(int i = 0; i < num_particles; i++) {
    auto num_gen = d(gen);
    new_particles.push_back(std::move(particles[num_gen]));
  }
  particles = std::move(new_particles);

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
