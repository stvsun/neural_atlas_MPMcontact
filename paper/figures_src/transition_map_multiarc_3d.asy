// Figure C: multi-arc contact in 3D.
// A two-footed rigid follower (a staple/bridge: flat top bar + two rounded feet)
// rests above a gently sin-modulated wavy sheet, touching it in TWO disjoint patches.
// Both patches are highlighted in red (arc 1, arc 2). A single teal circle on the
// deeper patch marks what a single closest-point projection would return (1 foot),
// contrasting the boundary scan (finds both) with closest-point (finds one).
settings.outformat = "pdf";
settings.prc      = false;
settings.render   = 8;        // smooth rasterized surfaces; vector labels/lines
settings.tex      = "latex";

import three;
import graph3;

usepackage("amsmath");
usepackage("amssymb");
usepackage("bm");

size(12.5cm, 0);
currentlight = light(gray(0.84), specular=gray(0.16),
                     (4,-5,8), (-5,2,5));
currentprojection = orthographic(camera=(3.0, -4.8, 3.4),
                                 up=Z, target=(0,0,0.15), zoom=0.92);

// ----- palette --------------------------------------------------------------
pen sheetfill  = rgb(0.94,0.82,0.62) + opacity(1.0);   // warm sandy obstacle
pen sheetmesh  = rgb(0.78,0.62,0.38) + linewidth(0.20);
pen footfill   = gray(0.66) + opacity(1.0);            // steel follower
pen barfill    = gray(0.70) + opacity(1.0);            // lighter top bar
pen arcpen     = rgb(0.82,0.12,0.10);                  // red contact patches
pen tealpen    = rgb(0.10,0.52,0.55) + linewidth(1.2); // closest-point marker
pen lblpen     = black + fontsize(9pt);
pen redlbl     = rgb(0.82,0.12,0.10) + fontsize(9pt);

// ----- the wavy obstacle sheet  z = w(x,y) ----------------------------------
real amp = 0.16;
real kx = 1.30, ky = 0.85;
real wsheet(real x, real y) {
  return amp*(sin(kx*x) + 0.55*cos(ky*y) + 0.35*sin(0.7*kx*x)*cos(0.9*ky*y));
}

real Xmin=-3.0, Xmax=3.0, Ymin=-1.7, Ymax=1.7;
surface sheet = surface(
  new triple(pair t){ return (t.x, t.y, wsheet(t.x,t.y)); },
  (Xmin,Ymin), (Xmax,Ymax), 64, 28, Spline);
draw(sheet, surfacepen=sheetfill, meshpen=nullpen);

// a sparse coordinate net on the sheet to read its undulation, without clutter
for (int i = 0; i <= 9; ++i) {
  real x = Xmin + (Xmax-Xmin)*i/9;
  path3 c = (x, Ymin, wsheet(x,Ymin));
  int M = 40;
  for (int k=1;k<=M;++k){ real y=Ymin+(Ymax-Ymin)*k/M; c=c--(x,y,wsheet(x,y)); }
  draw(c, sheetmesh);
}
for (int j = 0; j <= 5; ++j) {
  real y = Ymin + (Ymax-Ymin)*j/5;
  path3 c = (Xmin, y, wsheet(Xmin,y));
  int M = 60;
  for (int k=1;k<=M;++k){ real x=Xmin+(Xmax-Xmin)*k/M; c=c--(x,y,wsheet(x,y)); }
  draw(c, sheetmesh);
}

// ----- the two-footed follower (staple) -------------------------------------
// Two rounded feet at x = +/- footX; a flat top bar joining them.
real footX = 1.55;     // horizontal half-separation of the feet
real footR = 0.62;     // foot radius (rounded underside)
real footY = 0.0;

// foot centres chosen so each foot's lowest point grazes the sheet -> contact.
// the rigid bar is level, so both feet share the SAME bar height; whichever foot
// sits over a higher sheet crest penetrates more -> that is the "deeper" patch.
real barUnderside = 0.0;  // set below from the level bar
triple footCenter(real sx, real centreZ) { return (sx, footY, centreZ); }
// choose a single level so the LEFT foot (over the higher crest) presses deeper.
real zL = wsheet(-footX, footY);
real zR = wsheet( footX, footY);
real level = max(zL, zR) + footR + 0.05;   // feet just graze the higher crest
triple fcL = footCenter(-footX, level);
triple fcR = footCenter( footX, level);
real barZ  = level + footR*0.55;           // top bar sits across the feet tops

// draw the two feet as rounded caps (lower hemispheres + short cylindrical body)
void drawFoot(triple c) {
  // rounded underside: lower part of a sphere
  surface low = surface(
    new triple(pair t){
      real th=t.x, ph=t.y;
      return c + footR*(sin(th)*cos(ph), sin(th)*sin(ph), cos(th));
    }, (pi/2,0),(pi,2pi), 22, 30, Spline);   // theta in [pi/2,pi] -> lower half
  draw(low, surfacepen=footfill, meshpen=nullpen);
  // short cylindrical shank up to the bar
  surface shank = surface(
    new triple(pair t){
      real ph=t.x, h=t.y;
      return c + (footR*cos(ph), footR*sin(ph), h);
    }, (0,0),(2pi, barZ - c.z - 0.0), 30, 6, Spline);
  draw(shank, surfacepen=footfill, meshpen=nullpen);
}
drawFoot(fcL);
drawFoot(fcR);

// the flat top bar joining the two feet
real barH = 0.18, barW = 0.46, barOver = 0.30;
// build the bar as a box swept along x
path3 barBox(real z0){
  return (-footX-barOver, footY-barW, z0)--( footX+barOver, footY-barW, z0)
       --( footX+barOver, footY+barW, z0)--(-footX-barOver, footY+barW, z0)--cycle;
}
surface bartop = surface(barBox(barZ+barH));
draw(bartop, surfacepen=barfill, meshpen=nullpen);
// front and side faces of the bar
void barFace(triple a, triple b, triple cc, triple d){
  draw(surface(a--b--cc--d--cycle), surfacepen=barfill, meshpen=nullpen);
}
real bx0=-footX-barOver, bx1=footX+barOver, byo=footY-barW, byi=footY+barW;
barFace((bx0,byo,barZ),(bx1,byo,barZ),(bx1,byo,barZ+barH),(bx0,byo,barZ+barH));
barFace((bx0,byi,barZ),(bx1,byi,barZ),(bx1,byi,barZ+barH),(bx0,byi,barZ+barH));
barFace((bx0,byo,barZ),(bx0,byi,barZ),(bx0,byi,barZ+barH),(bx0,byo,barZ+barH));
barFace((bx1,byo,barZ),(bx1,byi,barZ),(bx1,byi,barZ+barH),(bx1,byo,barZ+barH));

// ----- the TWO contact patches on the sheet: identical clean red ellipses ----
// Each patch is a SINGLE closed red outline (a small flat ellipse) lying on the
// sheet beneath a foot's contact point.  Both are drawn the SAME way.  The patch
// centre is shifted toward the camera (-y) by patchFwd so the foot body does not
// occlude the near arc; it still sits squarely under its own foot in x.
real prx = 0.34, pry = 0.26;   // ellipse semi-axes (x wider, y foreshortened)
real patchFwd = 0.42;          // forward (toward-camera) shift so the ring clears the foot
real pcy = footY - patchFwd;   // patch centre in y
void contactEllipse(real sx) {
  int N = 72;
  path3 ell = (sx+prx, pcy, wsheet(sx+prx, pcy)+0.006);
  for (int k=1;k<=N;++k){
    real a = 2pi*k/N;
    real px = sx + prx*cos(a), py = pcy + pry*sin(a);
    ell = ell -- (px, py, wsheet(px,py)+0.006);
  }
  draw(ell, arcpen+linewidth(1.2));
}
contactEllipse(-footX);   // arc 1, under the LEFT foot
contactEllipse( footX);   // arc 2, under the RIGHT foot

// ----- arc labels: each leader aims at its own foot's ellipse ----------------
// arc 1 -> LEFT patch (label below-left); arc 2 -> RIGHT patch (label below-right).
// each dashed leader ends ON the front rim of its ellipse.
void arcLabel(string lab, real sx, pair side) {
  // tip on the near (camera-facing, -y) rim of the ellipse
  triple tip = (sx, pcy - pry, wsheet(sx, pcy - pry)+0.006);
  real dx = (side == W) ? -0.55 : 0.55;
  triple lpos = (sx + dx, Ymin - 0.55, 0.08);
  draw(lpos -- tip, arcpen+linewidth(0.5)+dashed);
  label(lab, lpos, align=side+0.6*S, redlbl);
}
arcLabel("arc 1", -footX, W);   // left patch, label sits below-left
arcLabel("arc 2",  footX, E);   // right patch, label sits below-right

// ----- single closest-point marker on the DEEPER (RIGHT) patch only (teal) ---
// The bar is level, so the foot over the HIGHER sheet crest presses deeper and is
// the global nearest point a single closest-point projection would return.
real sxDeep = footX;               // RIGHT foot (over the higher crest)
real tr = 0.155; int Nt=56;
real tyc = pcy;                    // sit on the same (visible) patch as arc 2
path3 tcirc = (sxDeep+tr, tyc, wsheet(sxDeep+tr,tyc)+0.012);
for (int k=1;k<=Nt;++k){
  real a=2pi*k/Nt; real px=sxDeep+tr*cos(a), py=tyc+tr*sin(a);
  tcirc = tcirc -- (px,py,wsheet(px,py)+0.012);
}
draw(tcirc, tealpen+linewidth(1.5));
dot((sxDeep, tyc, wsheet(sxDeep,tyc)+0.014), tealpen+linewidth(3.0));
// label parked in open sky ABOVE-right of the right foot (clear of the dark bar
// and well above the below-right "arc 2" label), with a dashed leader to the ring.
pen cptpen = rgb(0.06,0.38,0.41)+fontsize(8.5pt);
triple cptLabel = (sxDeep + 1.05, Ymin - 0.20, 1.65);
label("single closest-point: 1 foot", cptLabel, align=E, cptpen);
draw((sxDeep + 0.95, Ymin - 0.15, 1.52)
     -- (sxDeep + tr*0.3, tyc + tr*0.4, wsheet(sxDeep,tyc)+0.06),
     rgb(0.10,0.52,0.55)+linewidth(0.55)+dashed);

// invisible padding so labels are not clipped
dot((0,0,2.05), invisible);
dot((-3.4,0,0), invisible);
dot(( 3.4,0,0), invisible);
dot((0,2.2,0), invisible);
dot((0,-2.7,0), invisible);
