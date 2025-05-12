#include "utils.h"

using namespace std;

// solar parameters
double B0 = 3e3 * T2eV;                     // radiative zone max B [eV2]  200*T2eV;
double B1 = 50 * T2eV;                      // tachocline max B [eV2]  4*T2eV;//
double B2 = 3 * T2eV;                       // outer region max B [eV2]  3*T2eV;//
double r0 = 0.712 * rSolar;                 // [eV-1]
double r1 = 0.732 * rSolar;                 // [eV-1]
double d1 = 0.02 * rSolar;                  // [eV-1]
double r2 = 0.96 * rSolar;                  // [eV-1]
double d2 = 0.035 * rSolar;                 // [eV-1]
double ngamma0 = 1e25 * m2eV * m2eV * s2eV; // photon flux at r0 [eV3]

// solar model
vector<double> ne = read("data/ne.dat");       // electron number density [eV3]
vector<double> nbar = read("data/nbar.dat");   // Z2-summed number density [eV3]
vector<double> nbar2 = read("data/nbar2.dat"); // Z2-summed number density minus electrons [eV3]
vector<double> wp = read("data/wp.dat");       // plasma frequency [eV]
vector<double> T = read("data/T.dat");         // solar temperature [eV]
vector<double> r = read("data/r.dat");         // radial distance [eV-1]
vector<double> rho = read("data/rho.dat");     // solar density [eV4]
vector<double> nH = read("data/nH.dat");       // H number density [eV3]
vector<double> nHe3 = read("data/nHe3.dat");   // He3 number density [eV3]
vector<double> nHe4 = read("data/nHe4.dat");   // He4 number density [eV3]
// get gaunt factors
vector<vector<double>> z1 = readGaunt("data/Z1.dat"); // gaunt factors for Z=1
vector<vector<double>> z2 = readGaunt("data/Z2.dat"); // gaunt factors for Z=2

// parameters
double E = 2.4e-3;         // cham potential energy scale [eV] (2.4e-3 for cosmological const)
double n = 1;              // chameleon potential 1/phi^n
double Bm = 1e2;           // cham matter coupling
bool tachoclining = false; // for CAST comparison

// chameleon mass squared as a function of solar radius and model parameters
// cham model params n (phi-n potential), Bm (matter coupling)
// assume rho dominated by matter density
// units eV2
double mCham2(int c, double Bm)
{
    double E4n = pow(E, 4 + n);
    if (n < 0)
    {
        return n * (n + 1) * E4n * pow(pow(Bm * rho[c] / (n * Mpl * E4n), (n + 2)), 1 / (n + 1));
    }
    else
    {
        return n * (n + 1) * E4n * pow(Bm * rho[c] / (n * Mpl * E4n), (n + 2) / (n + 1));
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// ELECTRON-ION PRIMAKOFF ///////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

// integral I(u,v) solved analytically in cross section calc
// dimensionless, with dimensionless arguments
double curlyI(double u, double v)
{
    return (u * u - 1) / v * log((u - 1) / (u + 1)) - (pow(u + v, 2) - 1) / v * log((u + v - 1) / (u + v + 1)) - 2;
}

// I(u,v) in u=>1 limit
double curlyIapprox(double u, double v)
{ // for u->1
    return u * u / v - (v + 2) * log(v / (v + 2)) - 2;
}

// differential scalar production rate on earth d2N/dr/dw divided by beta_gamma^2
// transverse case
// units eV Bg-2
double T_integrand(int c, double Bm, double w)
{
    if (T[c] == 0)
    {
        return 0;
    } // solves weird behaviour when ne = T = 0
    double mg2 = 4 * pi * alpha * ne[c] / me; // assume mg2 = wp2
    double ms2 = mCham2(c, Bm);               // chameleon mass2 [eV2]
    // cout<<ms2<<endl;
    // double ms2 = Bm*Bm;						// fixed scalar mass2 [eV2]
    if (w * w <= mg2)
    {
        return 0;
    }
    if (w * w <= ms2)
    {
        return 0;
    }
    double K2 = 8 * pi * alpha * nbar[c] / T[c];             // Debye screening scale ^2 [eV2]
    double kgamma = sqrt(w * w - mg2);                       // photon momentum [eV]
    double kphi = sqrt(w * w - ms2);                         // scalar momentum [eV]
    double uArg = kgamma / (2 * kphi) + kphi / (2 * kgamma); // u for curlyI
    double vArg = K2 / (2 * kphi * kgamma);                  // v for curlyI
    double Iuv = curlyI(uArg, vArg);
    // explicitally put in the u=>1 limit to avoid badnesses in the code
    if (uArg < 1.01)
    {
        Iuv = curlyIapprox(uArg, vArg);
    }

    return alpha / (8 * Mpl * Mpl * pi) * pow(r[c], 2) * nbar[c] / (exp(w / T[c]) - 1) * w * w * kphi / kgamma * Iuv; // [eV Bg-2]
}

// integral over solar volume, for a given scalar mass and energy
// returns dN/dw Bg-2
// units Bg-2
double T_solarIntg(double w, double Bm)
{
    double total = 0;
    for (int c = 0; c < r.size() - 1; c++)
    {
        total += 0.5 * (r[c + 1] - r[c]) * (T_integrand(c + 1, Bm, w) + T_integrand(c, Bm, w));
    }
    return total; // [Bg-2]
}

// differential scalar production rate on earth d2N/dr/dw divided by beta_gamma^2
// transverse case, for line-of-sight ring integral
// units eV Bg-2
double T_integrand_ring(int c, double Bm, double w)
{
    if (T[c] == 0)
    {
        return 0;
    } // solves weird behaviour when ne = T = 0
    double mg2 = 4 * pi * alpha * ne[c] / me; // assume mg2 = wp2
    double ms2 = mCham2(c, Bm);               // chameleon mass2 [eV2]
    // cout<<ms2<<endl;
    // double ms2 = Bm*Bm;						// fixed scalar mass2 [eV2]
    if (w * w <= mg2)
    {
        return 0;
    }
    if (w * w <= ms2)
    {
        return 0;
    }
    double K2 = 8 * pi * alpha * nbar[c] / T[c];             // Debye screening scale ^2 [eV2]
    double kgamma = sqrt(w * w - mg2);                       // photon momentum [eV]
    double kphi = sqrt(w * w - ms2);                         // scalar momentum [eV]
    double uArg = kgamma / (2 * kphi) + kphi / (2 * kgamma); // u for curlyI
    double vArg = K2 / (2 * kphi * kgamma);                  // v for curlyI
    double Iuv = curlyI(uArg, vArg);
    // explicitally put in the u=>1 limit to avoid badnesses in the code
    if (uArg < 1.01)
    {
        Iuv = curlyIapprox(uArg, vArg);
    }

    return alpha / (8 * Mpl * Mpl * pi) * nbar[c] / (exp(w / T[c]) - 1) * w * w * kphi / kgamma * Iuv; // [eV Bg-2]
}

// integral over solar volume, for a line-of-sight ring
// returns dN/dw Bg-2
// ring inner and outer radii given as fraction of solar radius
// units Bg-2
double T_solarIntg_ring(double w, double Bm, double x)
{
    if (r[r.size() - 1] < x)
    {
        return 0;
    }
    double total = 0;
    int count = 0;
    for (int c = 0; c < r.size() - 3; c++)
    {
        if (r[c] < x)
        {
            count = c;
            continue;
        }
        total += 0.5 * (sqrt(pow(r[c + 1], 2) - x * x) * (T_integrand_ring(c + 2, Bm, w) - T_integrand_ring(c + 1, Bm, w)) + sqrt(pow(r[c], 2) - x * x) * (T_integrand_ring(c + 1, Bm, w) - T_integrand_ring(c, Bm, w)));
    }
    // cout << "count = "<<count<<"	r[count+1] = "<<r[count+1]<<"	x = "<<x<<endl;
    double boundary = sqrt(pow(r[r.size() - 1], 2) - x * x) * T_integrand_ring(r.size() - 1, Bm, w);
    // cout << boundary - total << endl;
    return boundary - total; // [Bg-2]
}

double T_solarIntg_x(double w, double Bm, double rmin, double rmax)
{
    double xmin = rmin * rSolar;
    double xmax = rmax * rSolar;
    double total = 0;
    double dx = 0.01 * rSolar;
    for (double x = xmin; x < xmax; x += dx)
    {
        total += 0.5 * (dx) * ((x + dx) * T_solarIntg_ring(w, Bm, x + dx) + x * T_solarIntg_ring(w, Bm, x));
    }
    if (total < 0)
    {
        cout << total << endl;
        return 0;
    }
    return total; // [Bg-2]
}

// calculate differential particle flux spectrum dN/dw by intg over solar volume, for line-of-sight ring
// units Bg-2
// (dN is really dN/dt, sorry)
void spectrum_ring()
{
    vector<double> count, energy;
    string name;
    Bm = 1e2; // cham matter coupling
    n = 1;    // cham model n
    double dw = 1e0;
    for (double w = dw; w < 2e4; w += dw)
    {
        energy.push_back(w);                              // eV
        count.push_back(T_solarIntg_x(w, Bm, 0.01, 1.0)); // Bg-2
        if ((int)(w) % (int)(1e3) == 0)
        {
            cout << "w = " << w / 1e3 << "keV of 20keV" << endl;
        }
    }
    // write to file
    name = "T_spectrum_ring_1to100.dat";
    write2D(name, energy, count);
}

int main()
{
    // convert Gaunt factor Theta to T in eV
    for (int i = 1; i < 201; i++)
    {
        z1[0][i] = z1[0][i] * me;
    }
    for (int i = 1; i < 201; i + // integral I(u,v) solved analytically in cross section calc
                                 // dimensionless, with dimensionless arguments
                             double curlyI(double u, double v) { return (u * u - 1) / v * log((u - 1) / (u + 1)) - (pow(u + v, 2) - 1) / v * log((u + v - 1) / (u + v + 1)) - 2; }

                             // I(u,v) in u=>1 limit
                             double curlyIapprox(double u, double v) { // for u->1
                                 return u * u / v - (v + 2) * log(v / (v + 2)) - 2;
                             } +)
    {
        z2[0][i] = z2[0][i] * me;
    }

    // spectrum('T');
    spectrum_ring();
    return 0;
}

// int main()
// {
//     double w = 1e3;  // test energy
//     double Bm = 100; // test coupling
//     double kernel = T_solarIntg(w, Bm);
//     cout << "kernel = " << kernel << endl;

//     double result = T_solarIntg_x(w, Bm, 0.1, 0.5);
//     cout << "dN/dw = " << result << endl;
//     return 0;
// }

// int main()
// {
//     double Bm = 1e2;    // chameleon matter coupling
//     double rmin = 0.01; // min radius (fraction of rSolar)
//     double rmax = 0.99; // max radius (fraction of rSolar)

//     double result = T_solarIntg_x(1000, Bm, 0.1, 0.9);
//     std::cout << "dN/dw = " << result << std::endl;

//     std::ofstream out("output_T_vs_E.dat");

//     for (double w = 10; w <= 1e4; w += 10)
//     { // sweep w from 1e-4 to 1e-2 eV
//         double result = T_solarIntg_x(w, Bm, rmin, rmax);
//         std::cout << w << " " << result << std::endl;
//         out << w << " " << result << std::endl;
//     }

//     out.close();
//     std::cout << "Data saved to output_T_vs_E.dat" << std::endl;

//     return 0;
// }
