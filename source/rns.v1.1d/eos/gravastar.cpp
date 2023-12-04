#include <fstream>
#include <string>
#include <cmath>
int main() {
    std::ofstream File("eosgravastar");
    long double max_energy = 3.2e16;
    long double max_log_energy = std::log10(max_energy);
    long double min_energy = 1.1e0;
    long double min_log_energy = std::log10(min_energy);
    int points = 102;
    long double energy_log_difference = (max_log_energy-min_log_energy)/points;

    for(int i=0; i<points; i++) {
        File << exp10(min_log_energy + energy_log_difference * i) << " " << 0.5*exp10(min_log_energy + energy_log_difference * i) << "\n";
    }
    File.close();
    return 0;
}