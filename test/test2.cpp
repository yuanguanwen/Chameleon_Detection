#include "utils.h"

#include <iostream>
#include <vector>
#include <tuple>

using namespace std;

// solar parameters
double B0 = 3000 * T2eV;                    // radiative zone max B [eV2]  200*T2eV;
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
/////////////////////////////////// MAGNETIC FIELD PRODUCTION /////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

// Gamma_photon
// simplified to only contain plasma and free-free effects
// units eV
double GammaPhoton(double w, int c, double g1, double g2)
{

    double p1 = 64 * pow(pi, 2) * pow(alpha, 3);
    double p2 = 3 * pow(me, 2) * pow(w, 3);
    double p3 = me * pow(ne[c], 2) / (2 * pi * T[c]);
    double p4 = 1 - exp(-w / T[c]);
    double p5 = 8 * pi * pow(alpha, 2) * ne[c] / (3 * pow(me, 2));

    // sum of ion densities
    double ions = (nH[c] * g1) + g2 * ((4 * nHe4[c]) + (4 * nHe3[c]));

    return p1 * pow(p2, -1) * pow(p3, 0.5) * p4 * ions + p5;
}

// solar B field
// units eV2
double Bfield(int c)
{
    // B field in solar radiative zone
    if (r[c] <= r0)
    {
        double l = (10 * (r0 / rSolar)) + 1;
        double K = (1 + l) * pow(1 + pow(l, -1), l) * B0;
        return K * pow(r[c] / r0, 2) * pow(1 - pow(r[c] / r0, 2), l);
    }
    // B-field in tachocline
    else if (r[c] > (r1 - d1) and r[c] < (r1 + d1))
    {
        return B1 * (1 - pow((r[c] - r1) / d1, 2));
    }
    // B-field in outer region
    else if (r[c] > (r2 - d2) and r[c] < (r2 + d2))
    {
        return B2 * (1 - pow((r[c] - r2) / d2, 2));
    }
    else
    {
        return 0;
    }
}

struct GResult
{
    double Gamma;
    double g1;
    double g2;
    int indexT1;
    int indexX1;
    int indexT2;
    int indexX2;
};

// selects Gaunt factor from matrix for Gamma
// returns Gamma [eV]
GResult selectG(int c, double w)
{
    // select g(w, T) value from matrix
    int indexT1;
    int indexT2;
    int indexX1;
    int indexX2;
    for (int i = 1; i < 200; i++)
    {
        if (z1[0][i] < T[c] and z1[0][i + 1] > T[c])
        {
            indexT1 = i;
        }
        if (z2[0][i] < T[c] and z2[0][i + 1] > T[c])
        {
            indexT2 = i;
        }
    }
    for (int i = 1; i < 500; i++)
    {
        if ((z1[i][0] * T[c]) < w and (z1[i + 1][0] * T[c]) > w)
        {
            indexX1 = i;
        }
        if ((z2[i][0] * T[c]) < w and (z2[i + 1][0] * T[c]) > w)
        {
            indexX2 = i;
        }
    }
    double g1 = z1[indexT1][indexX1];
    double g2 = z2[indexT2][indexX2];
    double Gamma = GammaPhoton(w, c, g1, g2);
    return {Gamma, g1, g2, indexT1, indexX1, indexT2, indexX2}; // std::make_tuple(gamma, g1, g2);
}

// differential scalar production rate d2N/dr/dw times Lambda2
// B-field conntribution
// units eV Bg-2
double B_integrand(int c, double Bm, double w)
{
    if (T[c] == 0)
    {
        return 0;
    } // solves weird behaviour when ne = T = 0
    double mg2 = 4 * pi * alpha * ne[c] / me; // assume mg2 = wp2 [eV2]
    double ms2 = mCham2(c, Bm);               // chameleon mass2 [eV2]
    // double ms2 = Bm*Bm;							// fixed scalar mass2 [eV2]
    if (w * w <= mg2)
    {
        return 0;
    }
    if (w * w <= ms2)
    {
        return 0;
    }
    double kgamma = sqrt(w * w - mg2); // photon momentum [eV]
    double kphi = sqrt(w * w - ms2);   // scalar momentum [eV]
    double B = Bfield(c);              // solar B field [eV2]

    GResult g = selectG(c, w);
    double G = g.Gamma;
    // std::tie(G, g1, g2) = selectG(c, w); // Unpack the tuple
    // std::tuple<double, double, double> selectG(c, w);
    // auto [G, g1, g2] = selectG(c, w);                                                                                                                   // Cleaner and safer
    return 2 / (pi * Mpl * Mpl) * pow(r[c], 2) * B * B * w * pow(w * w - ms2, 3 / 2) / (pow(ms2 - mg2, 2) + (w * w * G * G)) * G / (exp(w / T[c]) - 1); // [eV Bg-2]
}

// integral over solar volume, for a given scalar mass and energy
// B-field contribution, with option to fix to tachocline for comparison
// returns dN/dw Bg-2
// units Bg-2
double B_solarIntg(double w, double Bm)
{
    double total = 0;
    for (int c = 0; c < r.size() - 2; c++)
    {
        // tachoclining
        if (tachoclining)
        {
            if (r[c] / rSolar < r1 / rSolar - 0.05)
            {
                continue;
            }
            else if (r[c] / rSolar > r1 / rSolar + 0.05)
            {
                continue;
            }
        }
        total += 0.5 * (r[c + 1] - r[c]) * (B_integrand(c + 1, Bm, w) + B_integrand(c, Bm, w));
    }
    return total * 1e3 / s2eV;
}

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

// calculate differential particle flux spectrum dN/dw by intg over solar volume
// units Bg-2
// (dN is really dN/dt, sorry)
void spectrum(char option)
{
    vector<double> count, energy;
    string name;
    Bm = 1e2; // cham matter coupling
    n = 1;    // cham model n
    double dw = 1e0;
    for (double w = dw; w < 2e4; w += dw)
    {
        energy.push_back(w); // eV
        if (option == 'T')
        {
            count.push_back(T_solarIntg(w, Bm));
        } // Bg-2
        else if (option == 'B')
        {
            count.push_back(B_solarIntg(w, Bm));
        } // Bg-2
        if ((int)(w) % (int)(1e3) == 0)
        {
            cout << "w = " << w / 1e3 << "keV of 20keV" << endl;
        }
    }
    // write to file
    if (option == 'T')
    {
        name = "T_spectrum_1e2.dat";
    }
    else if (option == 'B')
    {
        name = "B_spectrum_1e2_low.dat";
    }

    write2D(name, energy, count);
}

// Assume the following functions are defined elsewhere and return appropriate types:
// double mCham2(int c, double Bm);
// double Bsolar(int c);
// double selectG(int c, double w);  // or a tuple-like return if g1/g2 needed
// double B_integrand(int c, double w, double Bm);
// double B_solarIntg(double w, double Bm);

void test_spectrum_debug(double Bm)
{
    vector<int> c_test_vals = {5, 50, 500, 1500};
    double w_test = 1000.0; // energy test in eV

    for (int c_test : c_test_vals)
    {
        cout << "mass [eV] = " << mCham2(c_test, Bm) << endl;
        cout << "Bsolar [eV] = " << Bfield(c_test) << endl;

        // Assuming selectG returns just Gamma, not individual g1/g2.
        // If it returns a struct/tuple of g1, g2, adjust accordingly.
        GResult g = selectG(c_test, w_test);
        cout << "GammaPhoton/g1/g2 = " << g.Gamma << ";" << g.g1 << ";" << g.g2 << endl;
        cout << "g1[x, y] = " << g.g1 << ";" << g.indexT1 << ";" << g.indexX1 << endl;
        cout << "g2[x, y] = " << g.g2 << ";" << g.indexT2 << ";" << g.indexX2 << endl;

        double kernel = B_integrand(c_test, w_test, Bm);
        cout << "kernel = " << kernel << endl;

        double spectrum = B_solarIntg(w_test, Bm);
        cout << "spectrum = " << spectrum << endl;

        // cout << "z1[164][310] = " << z1[164][310] << endl;
        // cout << "z1[510][510] = " << z1[500][500] << endl;
        // size_t numRows = z1.size();
        // size_t numCols = (numRows > 0) ? z1[0].size() : 0;
        // cout << "Shape: " << numRows << ", " << numCols << endl;

        cout << "===================================" << endl;
    }
}

int main()
{
    // convert Gaunt factor Theta to T in eV
    for (int i = 1; i < 201; i++)
    {
        z1[0][i] = z1[0][i] * me;
    }
    for (int i = 1; i < 201; i++)
    {
        z2[0][i] = z2[0][i] * me;
    }

    // saveGaunt(z1, "z1_saved.dat");
    // saveGaunt(z2, "z2_saved.dat");

    // spectrum('B');
    test_spectrum_debug(100.0);
    // spectrum_ring();
    return 0;
}
