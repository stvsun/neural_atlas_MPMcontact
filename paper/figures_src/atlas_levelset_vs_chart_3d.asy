// Figure A: ambient level set vs boundary chart, on the same non-convex body.
// Left  : translucent body + nested offset shells + medial axis (level-set view).
// Right : boundary surface only, with a (theta1,theta2) coordinate grid (chart view).
settings.outformat = "pdf";
settings.prc      = false;
settings.render   = 8;        // smooth surfaces rasterized at high res; labels/lines stay vector
settings.tex      = "latex";

import three;
import graph3;

usepackage("amsmath");
usepackage("amssymb");
usepackage("bm");

size(13cm, 0);
currentlight = light(gray(0.78), specular=gray(0.18),
                     (4,-3,6), (-4,2,5));
currentprojection = orthographic(camera=(3.7, -4.2, 1.9),
                                 up=Z, target=(0,0,0), zoom=1.05);

// ----- palette --------------------------------------------------------------
pen bodyfill   = gray(0.60) + opacity(0.42);
pen shellpen   = gray(0.46) + opacity(0.085);
pen surfteal   = rgb(0.18,0.52,0.55);     // teal accent for the chart
pen gridpen    = rgb(0.07,0.30,0.32) + linewidth(0.32);
pen medpen     = rgb(0.80,0.13,0.10) + linewidth(1.05) + dashed;
pen titlepen   = black + fontsize(10.5pt);
pen lblpen     = black + fontsize(9pt);
pen redlbl     = rgb(0.80,0.13,0.10) + fontsize(9pt);

// ----- the dumbbell radius r(theta,phi) -------------------------------------
// theta in [0,pi] (polar from body axis), phi in [0,2pi].  A pinch at the
// equator (theta=pi/2) makes the waist concavity that carries a medial axis.
real Rr(real theta) {
  real c = cos(theta);                      // along body axis
  real waist = 1.0 - 0.52*exp(-7.0*c*c);    // equatorial pinch
  return 0.98*waist;
}

// body axis tilted toward the viewer so the waist is clearly visible.
transform3 axisTilt = rotate(24, X) * rotate(-18, Y);

triple P(real theta, real phi, real scale, triple shift) {
  real r = scale*Rr(theta);
  triple loc = r*(sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta));
  return shift + axisTilt*loc;
}

surface bodySurface(real scale, triple shift, int nu=46, int nv=42) {
  return surface(
    new triple(pair t) { return P(t.x, t.y, scale, shift); },
    (0,0), (pi,2pi), nu, nv, Spline);
}

real titleZ = 1.78;   // common title height for both panels
triple cL = (-1.62, 0, 0);
triple cR = ( 1.62, 0, 0);

// ============================================================================
// LEFT PANEL : ambient level set
// ============================================================================

// two faint nested offset shells (suggest the scalar field phi in R^3)
draw(bodySurface(1.30, cL, 30, 28), surfacepen=shellpen, meshpen=nullpen);
draw(bodySurface(1.15, cL, 32, 30), surfacepen=shellpen, meshpen=nullpen);

// the body itself, translucent grey isosurface (the zero set boundary)
draw(bodySurface(1.00, cL), surfacepen=bodyfill, meshpen=nullpen);

// medial axis: the body-axis segment threading the waist where phi is
// non-smooth (a one-dimensional medial set).
triple ma = axisTilt*(0,0,1);
path3 medial = (cL - 0.50*ma) -- (cL + 0.50*ma);
draw(medial, medpen);
dot(cL - 0.50*ma, medpen+linewidth(2.4));
dot(cL + 0.50*ma, medpen+linewidth(2.4));
label("medial axis", cL + (0.20,0,-0.92), align=S, redlbl);

label("$\varphi:\mathbb{R}^3\!\to\!\mathbb{R}$", (cL.x, 0, titleZ), titlepen);

// ============================================================================
// RIGHT PANEL : boundary chart (surface only + coordinate grid)
// ============================================================================

surface chart = bodySurface(1.00, cR, 46, 44);
draw(chart, surfacepen=surfteal+opacity(0.96), meshpen=nullpen);

// (theta1,theta2) coordinate grid drawn ON the surface, lifted a hair outward
real lift = 1.006;
int nLines1 = 9;   // lines of constant theta
int nLines2 = 14;  // lines of constant phi
for (int i = 1; i < nLines1; ++i) {
  real th = pi*i/nLines1;
  int M = 72;
  path3 g = P(th, 0, lift, cR);
  for (int k = 1; k <= M; ++k) g = g -- P(th, 2pi*k/M, lift, cR);
  draw(g, gridpen);
}
for (int j = 0; j < nLines2; ++j) {
  real ph = 2pi*j/nLines2;
  int M = 56;
  path3 g = P(1e-3, ph, lift, cR);
  for (int k = 1; k <= M; ++k) g = g -- P(pi*k/M, ph, lift, cR);
  draw(g, gridpen);
}

// two coordinate-direction labels on a clear front-facing patch (upper lobe).
// Choose thL near the top so the meridian tangent points UP (away from waist).
real thL = pi*0.28, phL = -0.95;
triple base = P(thL, phL, lift, cR);
triple t1 = -unit(P(thL+0.10, phL, 1.0, cR) - P(thL, phL, 1.0, cR));  // -d/dtheta -> toward pole
triple t2 = unit(P(thL, phL+0.16, 1.0, cR) - P(thL, phL, 1.0, cR));   // d/dphi (parallel)
draw(base -- base + 0.46*t1, gray(0.10)+linewidth(0.75), Arrow3(HookHead3, size=4.5));
draw(base -- base + 0.46*t2, gray(0.10)+linewidth(0.75), Arrow3(HookHead3, size=4.5));
label("$\theta_1$", base + 0.66*t1, align=N, lblpen);
label("$\theta_2$", base + 0.66*t2, align=E, lblpen);

label("$\varphi:\Theta\!\to\!\partial\Omega$", (cR.x, 0, titleZ), titlepen);

// invisible padding dots so the auto crop keeps headroom for the titles
dot((cL.x, 0, titleZ+0.30), invisible);
dot((cR.x, 0, titleZ+0.30), invisible);
