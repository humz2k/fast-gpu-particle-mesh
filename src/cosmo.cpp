#include "cosmo.hpp"
#include "logging.hpp"
#include "common.hpp"
#include <assert.h>
#include <math.h>

PowerSpectrum::PowerSpectrum(const std::string& filename) {
    LOG_DEBUG("reading ipk %s", filename.c_str());

    double header[20];
    FILE* ptr = fopen(filename.c_str(), "rb");
    assert(ptr);

    assert(fread(header, sizeof(header), 1, ptr) == 1);

    m_k_min = header[0];
    m_k_max = header[1];
    m_k_bins = header[2];
    m_k_delta = header[3];

    m_h_values.resize(m_k_bins + 20);
    assert(fread(m_h_values.data(), sizeof(double), m_k_bins, ptr) ==
           (size_t)m_k_bins);

    fclose(ptr);
}

Cosmo::Cosmo(const Params& params)
    : m_params(params), m_initial_pk(m_params.ipk()) {}

static double da_dtau(double a, double OmM, double OmL){
    double da_dtau_2 = 1+OmM*((1/a)-1) + OmL*((a*a)-1);
    return sqrt(da_dtau_2);
}

static double da_dtau__3(double a, double OmM, double OmL){
    double da_dtau_1 = da_dtau(a,OmM,OmL);
    return 1/(da_dtau_1*da_dtau_1*da_dtau_1);
}

static double int_1_da_dtau_3(double a, double OmM, double OmL, int bins){
    double start = 0;
    double end = a;
    double delta = (end-start)/((double)bins);
    double sum = 0;
    for (int k = 1; k < bins; k++){
        sum += da_dtau__3(start + ((double)k)*delta,OmM,OmL);
    }
    sum += (da_dtau__3(start,OmM,OmL) + da_dtau__3(end,OmM,OmL))/2.0f;
    sum *= delta;
    return sum;
}

static double calc_delta(double a, double OmM, double OmL){
    double integral = int_1_da_dtau_3(a,OmM,OmL,100);
    double diff = da_dtau(a,OmM,OmL);
    double mul = (5*OmM)/(2*a);
    return mul*diff*integral;
}

static double calc_dot_delta(double a, double OmM, double OmL, double h){
    return (calc_delta(a+h,OmM,OmL)-calc_delta(a-h,OmM,OmL))/(2.0f*h);
}

double Cosmo::delta(double z) const{
    double OmM = m_params.omega_matter();
    double OmL = 1.0 - (OmM + m_params.omega_nu());
    return calc_delta(z2a(z),OmM,OmL);
}

double Cosmo::dot_delta(double z) const{
    double OmM = m_params.omega_matter();
    double OmL = 1.0 - (OmM + m_params.omega_nu());
    return calc_dot_delta(z2a(z),OmM,OmL,0.0001);
}

void Cosmo::rkck(double* y, double* dydx, int n, double x, double h,
		      double* yout, double* yerr,
		      void (Cosmo::*derivs)(double, double*, double*))
{
  int i;
  static double a2=0.2,a3=0.3,a4=0.6,a5=1.0,a6=0.875,b21=0.2,
    b31=3.0/40.0,b32=9.0/40.0,b41=0.3,b42 = -0.9,b43=1.2,
    b51 = -11.0/54.0, b52=2.5,b53 = -70.0/27.0,b54=35.0/27.0,
    b61=1631.0/55296.0,b62=175.0/512.0,b63=575.0/13824.0,
    b64=44275.0/110592.0,b65=253.0/4096.0,c1=37.0/378.0,
    c3=250.0/621.0,c4=125.0/594.0,c6=512.0/1771.0,
    dc5 = -277.00/14336.0;
  double dc1=c1-2825.0/27648.0,dc3=c3-18575.0/48384.0,
    dc4=c4-13525.0/55296.0,dc6=c6-0.25;
  double *ak2,*ak3,*ak4,*ak5,*ak6,*ytemp;

  ak2= (double *)malloc(n*sizeof(double));
  ak3= (double *)malloc(n*sizeof(double));
  ak4= (double *)malloc(n*sizeof(double));
  ak5= (double *)malloc(n*sizeof(double));
  ak6= (double *)malloc(n*sizeof(double));
  ytemp= (double *)malloc(n*sizeof(double));

  for (i=0; i<n; ++i)
    ytemp[i]=y[i]+b21*h*dydx[i];
  (this->*derivs)(x+a2*h,ytemp,ak2);
  for (i=0; i<n; ++i)
    ytemp[i]=y[i]+h*(b31*dydx[i]+b32*ak2[i]);
  (this->*derivs)(x+a3*h,ytemp,ak3);
  for (i=0; i<n; ++i)
    ytemp[i]=y[i]+h*(b41*dydx[i]+b42*ak2[i]+b43*ak3[i]);
  (this->*derivs)(x+a4*h,ytemp,ak4);
  for (i=0; i<n; ++i)
    ytemp[i]=y[i]+h*(b51*dydx[i]+b52*ak2[i]+b53*ak3[i]+b54*ak4[i]);
  (this->*derivs)(x+a5*h,ytemp,ak5);
  for (i=0; i<n; ++i)
    ytemp[i]=y[i]+h*(b61*dydx[i]+b62*ak2[i]+b63*ak3[i]+b64*ak4[i]+b65*ak5[i]);
  (this->*derivs)(x+a6*h,ytemp,ak6);
  for (i=0; i<n; ++i)
    yout[i]=y[i]+h*(c1*dydx[i]+c3*ak3[i]+c4*ak4[i]+c6*ak6[i]);
  for (i=0; i<n; ++i)
    yerr[i]=h*(dc1*dydx[i]+dc3*ak3[i]+dc4*ak4[i]+dc5*ak5[i]+dc6*ak6[i]);

  free(ytemp);
  free(ak6);
  free(ak5);
  free(ak4);
  free(ak3);
  free(ak2);
}

#define SAFETY 0.9
#define PGROW -0.2
#define PSHRNK -0.25
#define ERRCON 1.89e-4
static double maxarg1,maxarg2, minarg1, minarg2;
#define FMAX(a,b) (maxarg1=(a),maxarg2=(b),(maxarg1) > (maxarg2) ? (maxarg1) : (maxarg2))
#define FMIN(a,b) (minarg1=(a),minarg2=(b),(minarg1) < (minarg2) ? (minarg1) : (minarg2))
void Cosmo::rkqs(double* y, double* dydx, int n, double *x, double htry,
		      double eps,
                      double* yscal, double *hdid, double *hnext, int *feval,
                      void (Cosmo::*derivs)(double, double *, double *))
{
  int i;
  double errmax,h,htemp,xnew,*yerr,*ytemp;

  yerr= (double *)malloc(n*sizeof(double));
  ytemp= (double *)malloc(n*sizeof(double));
  h=htry;

  for (;;) {
    rkck(y,dydx,n,*x,h,ytemp,yerr,derivs);
    *feval += 5;
    errmax=0.0;
    for (i=0; i<n; ++i) {errmax=FMAX(errmax,fabs(yerr[i]/yscal[i]));}
    errmax /= eps;
    if (errmax <= 1.0) break;
    htemp=SAFETY*h*pow((double) errmax,PSHRNK);
    h=(h >= 0.0 ? FMAX(htemp,0.1*h) : FMIN(htemp,0.1*h));
    xnew=(*x)+h;
    if (xnew == *x) {
      LOG_ERROR("Stepsize underflow in ODEsolve rkqs");
      exit(1);
    }
  }
  if (errmax > ERRCON) *hnext=SAFETY*h*pow((double) errmax,PGROW);
  else *hnext=5.0*h;
  *x += (*hdid=h);
  for (i=0; i<n; ++i) {y[i]=ytemp[i];}
  free(ytemp);
  free(yerr);
}
#undef SAFETY
#undef PGROW
#undef PSHRNK
#undef ERRCON
#undef FMAX
#undef FMIN

#define MAXSTP 10000
#define TINY 1.0e-30
#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))
void Cosmo::odesolve(double* ystart, int nvar, double x1, double x2,
			  double eps, double h1,
                          void (Cosmo::*derivs)(double, double *, double *),
			  bool print_stat)
{
  int i, nstp, nok, nbad, feval;
  double x,hnext,hdid,h;
  double *yscal,*y,*dydx;
  const double hmin=0.0;

  feval = 0;
  yscal= (double *)malloc(nvar*sizeof(double));
  y= (double *)malloc(nvar*sizeof(double));
  dydx= (double *)malloc(nvar*sizeof(double));

  x=x1;
  h=SIGN(h1,x2-x1);
  nok = nbad = 0;
  for (i=0; i<nvar; ++i) {y[i]=ystart[i];}

  for (nstp=0; nstp<MAXSTP; ++nstp) {
    (this->*derivs)(x, y, dydx);
    ++feval;
    for (i=0; i<nvar; ++i)
    {yscal[i]=fabs(y[i])+fabs(dydx[i]*h)+TINY;}
    if ((x+h-x2)*(x+h-x1) > 0.0) h=x2-x;
    rkqs(y,dydx,nvar,&x,h,eps,yscal,&hdid,&hnext,&feval,derivs);
    if (hdid == h) ++nok; else ++nbad;
    if ((x-x2)*(x2-x1) >= 0.0) {
      for (i=0; i<nvar; ++i) {ystart[i]=y[i];}
      free(dydx);
      free(y);
      free(yscal);
      if (print_stat){
	LOG_INFO("ODEsolve:\n");
	LOG_INFO(" Evolved from x = %f to x = %f\n", x1, x2);
	LOG_INFO(" successful steps: %d\n", nok);
	LOG_INFO(" bad steps: %d\n", nbad);
	LOG_INFO(" function evaluations: %d\n", feval);
      }
      return;
    }
    if (fabs(hnext) <= hmin) {
      LOG_ERROR("Step size too small in ODEsolve");
      exit(1);
    }
    h=hnext;
  }
  LOG_ERROR("Too many steps in ODEsolve");
  exit(1);
}
#undef MAXSTP
#undef TINY
#undef SIGN

double Cosmo::Omega_nu_massive(double a) {
  double mat = m_params.omega_nu()/pow(a,3.0f);
  double rad = m_params.f_nu_massive()*m_params.omega_radiation()/pow(a,4.0f);
  return (mat>=rad)*mat + (rad>mat)*rad;
}

void Cosmo::growths(double a, double* y, double* dydx) {
  double H;
  H = sqrt(m_params.omega_cb()/pow(a,3.0)
	   + (1.0 + m_params.f_nu_massless())*m_params.omega_radiation()/pow(a,4.0)
	   + Omega_nu_massive(a)
	   + (1.0 - m_params.omega_matter() - (1.0+m_params.f_nu_massless())*m_params.omega_radiation()
	   * pow(a,(-3.0*(1.0+m_params.w_de()+m_params.wa_de())))*exp(-3.0*m_params.wa_de()*(1.0-a))
      ));
  dydx[0] = y[1]/(a*H);
  dydx[1] = -2.0*y[1]/a + 1.5*m_params.omega_cb()*y[0]/(H*pow(a, 4.0f));
}

void Cosmo::update_growth_factor(double z){
    double x1, x2, dplus, ddot;
  const double zinfinity = 100000.0;

  x1 = 1.0/(1.0+zinfinity);
  x2 = 1.0/(1.0+z);
  double ystart[2];
  ystart[0] = x1;
  ystart[1] = 0.0;

  odesolve(ystart, 2, x1, x2, 1.0e-6, 1.0e-6, &Cosmo::growths, false);
  //printf("Dplus = %f;  Ddot = %f \n", ystart[0], ystart[1]);

  dplus = ystart[0];
  ddot  = ystart[1];
  x1 = 1.0/(1.0+zinfinity);
  x2 = 1.0;
  ystart[0] = x1;
  ystart[1] = 0.0;

  odesolve(ystart, 2, x1, x2, 1.0e-6, 1.0e-6, &Cosmo::growths, false);
  //printf("Dplus = %f;  Ddot = %f \n", ystart[0], ystart[1]);

  m_gf    = dplus/ystart[0];
  m_g_dot = ddot/ystart[0];
  m_last_growth_z = z;
}