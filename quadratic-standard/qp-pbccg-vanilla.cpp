//===================================================
// Location transportation problem
// Authors: Débora Ribeiro - deboralvesribeiro@gmail.com 
//          Ricardo Camargo - risargo@gmail.com
//          Gilberto Miranda - gilbertomirandajr@gmail.com
// PBCCG- Proximal Bundle Column-and-Constraint generation 
//===================================================
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cmath>
#include <sstream>
#include <algorithm>

using namespace std;

#include <ilcplex/ilocplex.h>
//#include <ilcplex/ilocplexi.h>
#define EPSILON 1e-8
//#define TIMELIMIT 518400
ILOSTLBEGIN
// ==============================================
// data structure 
// ==============================================
typedef struct{
    int m;                         // potential facilities
    int n;                         // customers
    int nm;
    vector <int> f;                // fixed costs f_i
    vector <int> a;                // capacity costs a_i
    vector <int> dd;               // demands d_j
    vector <int> dde;              // demand deviations _d_j
    vector <vector<double> > c;    // transportation costs c_ij
    vector <int> k;                // maximal allowed capacity k_i
    vector < double> ubb1; 
    vector < double> lbb1; 
    vector < double> timeN1; 
    vector < double> ubb2; 
    vector < double> lbb2; 
    vector < double> timeN2;
    double Gamma;                  // uncertainty budget 
    double ex;
    char name[250];
    vector <vector<double> > _e;  // erro _e
} DAT;
// ==============================================
// struct of mp and sp
// ==============================================
typedef struct{ 
    IloNum sup;
    IloNum ub;
    IloNum lb;
    IloNum _lb;
    IloNum gap;
    IloNum maxD;
    IloNum it;
    IloNum iter;
    IloNum nobb;
    IloNum Timeprob;
    IloNum TimeMaster;
    IloTimer *crono;
    IloEnv env;                            
    IloCplex cplex;
    IloModel mod;                          
    IloNumVarArray x;
    IloNumVarArray z;
    IloNumVarArray y;
//    IloNumVarArray xsi;
    IloNumVarArray e;
    IloNumVar eta;
    IloNumArray _z;
    IloNumArray _ze;    
    IloNumArray _y;
    IloObjective mpof;
    IloRangeArray desvioe;
    IloRangeArray constraints;
    IloRangeArray constraintssystems1;
    IloRangeArray constraintssystems2;
    IloRangeArray constraintssystems4;
    IloNumArray rhsd; 
    IloNum fat;
    IloNum rhs;
    IloNumArray _lbsdesvioe;    
    IloNumArray _rbsdesvioe;
    IloNum IsCCG;
    IloNum IsBundle;
    IloNum MpInfeasible;
    IloNum flag;
    IloNum Lk;
    IloNumVarArray vars;
    IloNumArray _vars;
    IloNum _tk;
    IloNum _tk0;
    IloNum _count;
} MP_CPX_DAT;	

typedef struct{ 
    IloTimer *crono;
    IloEnv env;                            
    IloCplex cplex;
    IloModel mod;                          
    IloNumVarArray w;
    IloNumVarArray g;
    IloNumVarArray pi;
    IloNumVarArray lambda;
    IloNumArray coefpi;
    IloNumArray coefg;
    IloNumArray _g;
    IloRangeArray constraints;
    IloObjective spof;
    IloObjective spom;
    IloNum _spof;
    IloNumArray BigM;
    IloNumArray _pi;
} SP_CPX_DAT;

void help();
void read_data(char name[],DAT &d);   
void create_model_mp(DAT &d, MP_CPX_DAT & mp);
void create_model_spM(DAT &d, SP_CPX_DAT &spM);
void create_model_sp(DAT &d, SP_CPX_DAT &sp, SP_CPX_DAT &spM);
void solve_mp(DAT &d, MP_CPX_DAT & mp);
void solve_spM(DAT &d, SP_CPX_DAT &spM, MP_CPX_DAT & mp);
void solve_sp(DAT &d, SP_CPX_DAT &sp,SP_CPX_DAT &spM, MP_CPX_DAT & mp);
void add_constraintssystems(DAT &d, MP_CPX_DAT & mp);
void iniciatilization(DAT &d, MP_CPX_DAT & mp);
void Print_solution(DAT &d,MP_CPX_DAT &mp, SP_CPX_DAT &sp);
// ============================================================================================

// main program
// ============================================================================================
int main (int argc, char *argv[]){
    // ===============
    // dat
    // ===============
    DAT d;
    if (argc < 2){
        help();
    }
    double Lk = (argc > 2) ? atof(argv[2]) : 1.0;
    read_data(argv[1],d);
    int nh = 0; int nobb = 0;   double TimeMasterInicial = 0;
    // ===============
    // cplex environment
    // ===============
    MP_CPX_DAT mp;
    SP_CPX_DAT spM;
    SP_CPX_DAT sp;
    mp.Lk = Lk ;
    try {
        //printf("\nLocation transportation model \n\n");
        //printf("\nCreating model...\n");   
        IloTimer crono(mp.env);
        mp.crono = &crono; 
        iniciatilization(d, mp);
        create_model_mp(d, mp);
        create_model_spM(d, spM);      
        create_model_sp(d, sp, spM);                   
        // ===============
        // main loop
        // ===============      
        crono.start();
        while(mp.gap > EPSILON and crono.getTime() <= 21600){
            if (mp.IsCCG == 1 && mp.IsBundle == 0){
                //  //cout << "\n    ** C&CG ** " << endl;
                // ===========
                // mp
                // ===========
                TimeMasterInicial = crono.getTime();
                solve_mp(d, mp);
                mp.TimeMaster = mp.TimeMaster + (crono.getTime() - TimeMasterInicial);
                nobb = nobb + mp.cplex.getNnodes();              
                // ===========
                // sp
                // ===========
                solve_spM(d, spM, mp);
                solve_sp(d,sp,spM, mp);
                // ===========
                // 
                // ===========                
                ++mp.it;
                add_constraintssystems(d, mp); 
            }
            else{
                //  //cout << "\n   ** Proximal bundle C&CG ** " << endl;
                // ===========
                // mp
                // ===========
                mp.flag == 1;
                TimeMasterInicial = crono.getTime();
                solve_mp(d, mp);
                mp.TimeMaster = mp.TimeMaster + (crono.getTime() - TimeMasterInicial);
                nobb = nobb + mp.cplex.getNnodes();

                // ===========
                // sp
                // ===========
                if (mp.MpInfeasible != 1){      
                    solve_spM(d, spM, mp);
                    solve_sp(d,sp,spM, mp);
                    ++mp.it;
                    add_constraintssystems(d, mp); 
                }
            }
            ++mp.iter;
            mp.gap = (mp.ub - mp.lb)/mp.ub;
            ++nh;
            printf ("%5d |%5d | %8.2f | %8.2f | %8.4f | %8.2f | bdl: %5.1f | ccg: %5.1f | inf: %5.1f \n",nh , nobb, mp.lb, mp.ub, 100*mp.gap, crono.getTime(), mp.IsBundle, mp.IsCCG, mp.MpInfeasible);
        }//endWhile
        
        mp.Timeprob = crono.getTime();
        mp.nobb = nobb; 
        Print_solution(d,mp,sp);
    }//endTry 

    catch (IloException& ex) {                      
    cerr << "Error: " << ex << endl;
    }
    return 0;
} //end main

// ============================================================================================
// ============================================================================================

// ========================================================
// function
// ========================================================
void help(){
  cout << endl << endl << "exec [data file] [ex fixed costs] \n " << endl;
  exit(1);
}
// ==============================
// read dat
// ==============================
void read_data(char name[], DAT &d){   
    ifstream arq(name);
    if (!arq.is_open()){
            help();
    }
    strcpy(d.name, name);
    arq >> d.n;
    arq >> d.m;
    d.nm = d.n * d.m;
    arq >> d.Gamma;
    d.f = vector<int>(d.m + 1,0.0);
    d.a = vector<int>(d.m + 1,0.0);
    d.k = vector<int>(d.m + 1,0.0);
    d.dd = vector<int>(d.n + 1);
    d.dde = vector<int>(d.n + 1);
    d.c = vector<vector <double > >(d.m + 1,vector<double>(d.n +1));
    d._e= vector<vector<double > >(50,vector<double>(d.m));       
    for (int j = 1; j <= d.n; ++j){
        arq >> d.dd[j];
        arq >> d.dde[j];
    }
        // // fixed costs, capacity costs e capacity of the facility
    for (int i = 1; i <= d.m; ++i){
        arq >> d.f[i];
        arq >> d.a[i];
        arq >> d.k[i];
    }   
        // // transportation costs;
    for (int i = 1; i <= d.m; ++i){
        for (int j = 1; j <= d.n; ++j){
            arq >> d.c[i][j];
        }
    }
        // // erro costs;
    double fat = 1.00e+07;
    for(int hh = 1; hh <= 50; ++hh){
        for(int i = 1; i <= d.m; ++i){    
                d._e[hh-1][i-1] = fat;
                arq >> d._e[hh-1][i-1];
                //cout << "   ** d._e[hh-1][i-1] ** " << d._e[hh-1][i-1]<< endl;
        }
        fat = 0.5*fat;
    }
    arq.close();
}
// ==============================
// iniciatilization
// ==============================
void iniciatilization(DAT &d, MP_CPX_DAT & mp){
    mp.ub = 1.00e+12;  mp.lb = 0.00e+00;  mp.it = 0;  mp.sup = 0;  mp.gap = 1; mp.TimeMaster = 0;                  
    mp.IsCCG = 0;   mp.IsBundle = 1;  mp.iter = 0;  mp._count = -1;   mp._tk = 1;  
}
// ==============================
// Function mp problem
// ==============================
void create_model_mp(DAT &d, MP_CPX_DAT & mp){
    IloEnv env = mp.env;
    mp.mod = IloModel(env); 
    mp.cplex = IloCplex(mp.mod);   
    mp.cplex.setWarning(env.getNullStream());
    mp.cplex.setOut(env.getNullStream());
    mp.cplex.setParam(IloCplex::Threads,1);
    mp.cplex.setParam(IloCplex::EpGap, 1.00e-08);        //mipgap
    mp.cplex.setParam(IloCplex::MIPDisplay, 2);          //mipdisplay
    //=========================
    //variables  
    //========================= 
    mp.vars = IloNumVarArray(env);
    mp._vars = IloNumArray(env);
    mp.x = IloNumVarArray(env); 
//    mp.xsi = IloNumVarArray(env,d.m,0.0,+IloInfinity,ILOFLOAT);   
    mp.y = IloNumVarArray(env,d.m,0.0,1.0,ILOINT);
    mp._y = IloNumArray(env,d.m);
    mp.z = IloNumVarArray(env,d.m,0.0,+IloInfinity,ILOFLOAT);
    mp._z = IloNumArray(env,d.m);  
    mp.e = IloNumVarArray(env,d.m, -IloInfinity,+IloInfinity,ILOFLOAT); 
    mp._ze = IloNumArray(env,d.m);
    mp.rhsd = IloNumArray(env,d.n);
    mp.eta = IloNumVar(env,0.0,+IloInfinity,ILOFLOAT);    
    mp.constraints = IloRangeArray(env);
    mp.desvioe = IloRangeArray(env);
    mp.constraintssystems1 = IloRangeArray(env);
    mp.constraintssystems2 = IloRangeArray(env);
    mp.constraintssystems4 = IloRangeArray(env);
    mp._rbsdesvioe = IloNumArray(env);
    mp._lbsdesvioe = IloNumArray(env);
    char name[50];

//===============
//===============
    mp.maxD = 0.0 ;
    for(int j = 1; j <= d.n; ++j){
        mp.maxD += d.dd[j]+d.dde[j];
    }
    for(int i = 1; i <= d.m; ++i){
        mp._ze[i-1] = mp.maxD/d.n;
    }
     
//===============
//obj. function- mpof
//===============
        //  // minimize mpof:  a0 * (1/(2tk)*sum{i in I}xsi[i] )  +  (eta); 
    IloExpr xpfo(env);
    xpfo +=  mp.eta;
    double coefx = 1/(2*mp._tk);    
    for(int i = 1; i <= d.m; ++i){
       //xpfo += mp.IsBundle * mp.xsi[i - 1]; 
       xpfo += mp.IsBundle * coefx * mp.e[i - 1] * mp.e[i - 1] ; 
    }
    mp.mpof = IloAdd(mp.mod, IloMinimize(env, xpfo));
    xpfo.end();     
    //===============
    //constraint
    //===============   
        //  //mpmpfeasibility: sum{i in I} z[i] >=  _maxD;    
    IloExpr r1mpfeasibility(env);
    for(int i = 1; i <= d.m; ++i){
        r1mpfeasibility += mp.z[i - 1];
    }
        r1mpfeasibility -= mp.maxD;                      
        mp.constraints.add(r1mpfeasibility >= 0.0);
        r1mpfeasibility.end();                                   
           
        //  //mpmpcapacity{i in I}: z[i] <= 6*k[i] * y[i];   
    for(int i = 1; i <= d.m; ++i){
        IloExpr r2mpcapacity(env);
        r2mpcapacity += mp.z[i - 1];               
        r2mpcapacity -= 6*d.k[i]*mp.y[i - 1]; 
        mp.constraints.add(r2mpcapacity <= 0.0);
        r2mpcapacity.end();                                   
    }     
        //  //rdesvioe{i in I}: e[i] = (z[i] - _ze[i]);
   for(int i = 1; i <= d.m; ++i){
        IloExpr r3rdesvioe(env);
        r3rdesvioe += mp.e[i - 1];
        r3rdesvioe -= mp.z[i - 1];
        mp.desvioe.add( -1.0*mp._ze[i - 1] <= r3rdesvioe <= -1.0*mp._ze[i - 1]);
        r3rdesvioe.end(); 
        
        mp._rbsdesvioe.add(-1.0*mp._ze[i - 1]); 
        mp._lbsdesvioe.add(-1.0*mp._ze[i - 1]);
    }    
    /*    //  //oac{i in I, hh in 1..K}: xsi[i] >= _e[hh,i] * _e[hh,i] + 2.0 * (_e[hh,i]) * (e[i] - _e[hh,i]); 
    for(int i = 1; i <= d.m; ++i){
        for(int hh = 1;  hh <= 50; hh++){
            IloExpr r4oac(env);
            r4oac += mp.xsi[i - 1];               
            r4oac -= d._e[hh-1][i-1]*d._e[hh-1][i-1];
            r4oac -= 2*(d._e[hh-1][i-1])*(mp.e[i-1] - d._e[hh-1][i-1]);
            mp.constraints.add(r4oac >= 0.0);
            r4oac.end();  
        }                                 
    }*/
    
    mp.mod.add(mp.constraints); 
    mp.mod.add(mp.desvioe); 
}
// ==============================
//   constraints  - mp 
// ==============================
void add_constraintssystems(DAT &d, MP_CPX_DAT &mp){    
    IloEnv env = mp.cplex.getEnv();
    register int i;
    register int j;
        //  //mpsupply{h in H, i in I}: sum{j in J} x[h,i,j] <= z[i];
    for ( i = 1; i <= d.m; ++i){
        IloExpr r41mpsupply(env);
        r41mpsupply -= mp.z[i - 1];
        mp.constraintssystems1.add(r41mpsupply <= 0);        
        r41mpsupply.end();
    }
        //  //mpdemand{h in H, j in J}: sum{i in I} x[h,i,j] >= rhsd[h,j];
    for ( j = 1; j <= d.n; ++j){
        IloExpr r51mpdemand(env);
        r51mpdemand -= mp.rhsd[j - 1];   
        mp.constraintssystems2.add(r51mpdemand >= 0);
        r51mpdemand.end(); 
    }
    
        //  //mpcut2{h in H}: sum{i in I} 10*f[i] * y[i] + sum{i in I} a[i] * z[i] + sum{i in I, j in J} c[i,j] * x[h,i,j] <= eta;
    IloExpr r71mpcut(env);
    for (int i = 1; i <= d.m; ++i){
        r71mpcut += 10*d.f[i] * mp.y[i - 1];
        r71mpcut += d.a[i] * mp.z[i - 1];
    }
    r71mpcut -= mp.eta;
    mp.constraintssystems4.add(r71mpcut <= 0);
    r71mpcut.end();
    
    mp.mod.add(mp.constraintssystems1);   
    mp.mod.add(mp.constraintssystems2);   
    mp.mod.add(mp.constraintssystems4);  
    // ===============   
    // function Add x
    // ===============   
    int cont = 0; 
    for ( i = 1; i <= d.m; ++i){
        for ( j = 1; j <= d.n; ++j){
            IloNumColumn newCol = mp.mpof (0.00);
            newCol += mp.constraintssystems1[(mp.it-1)*(d.n) + i-1](1.00);
            newCol += mp.constraintssystems2[(mp.it-1)*d.m + j-1](1.00);
            newCol += mp.constraintssystems4[mp.it-1](d.c[i][j]);

            IloNumVar newVar(newCol, 0.0, +IloInfinity,ILOFLOAT);
            mp.mod.add(newVar);
            mp.x.add(newVar);
        }
    }
}
// ==============================
// Function spM problem
// ==============================
void create_model_spM(DAT &d, SP_CPX_DAT &spM){
    IloEnv env = spM.env;
    spM.mod = IloModel(env); 
    spM.cplex = IloCplex(spM.mod);
    spM.cplex.setWarning(env.getNullStream());
    spM.cplex.setOut(env.getNullStream());
    spM.cplex.setParam(IloCplex::Threads,1);
    spM.cplex.setParam(IloCplex::EpGap, 1.00e-08);        //mipgap
    spM.cplex.setParam(IloCplex::MIPDisplay, 2);          //mipdisplay
    //=========================
    //variables  
    //=========================   
    spM.pi = IloNumVarArray(env,d.m,0.0,+IloInfinity,ILOFLOAT);     
    spM.coefpi = IloNumArray(env,d.m);
    spM.lambda = IloNumVarArray(env,d.n,0.0,+IloInfinity,ILOFLOAT);
    spM.BigM = IloNumArray(env,d.n);  
    spM.constraints = IloRangeArray(env);
    //===============
    //obj. function- spoM
    //===============
        //  //maximize spoM: sum{j in J}( ( d[j] + d_[j] ) * lambda[j] ) - sum{i in I}( _z[i] * pi[i] );
    IloExpr xpfo(env);
    for(int j = 1; j <= d.n; ++j){   
        xpfo += (d.dd[j]+d.dde[j])*spM.lambda[j-1];
    }
    for(int i = 1; i <= d.m; ++i){
        xpfo -= spM.pi[i-1]; 
    }
    spM.spom = IloAdd(spM.mod, IloMaximize(env, xpfo));
    xpfo.end();
    //===============
    //constraint
    //===============   
        //  //spr1{i in I, j in J}: lambda[j] - pi[i] <= c[i,j]; 
    for(int i = 1;i <= d.m; ++i){
        for(int j = 1; j <= d.n; ++j){
        IloExpr sprm1(env);  
        sprm1 += spM.lambda[j-1];
        sprm1 -= spM.pi[i-1];       
        sprm1 -= d.c[i][j];
        spM.constraints.add(sprm1 <= 0.0);
        sprm1.end();            
        }
    }
    spM.mod.add(spM.constraints);        
}
// ==============================
// Function sp problem
// ==============================
void create_model_sp(DAT &d, SP_CPX_DAT &sp, SP_CPX_DAT &spM){
    IloEnv env = sp.env;
    sp.mod = IloModel(env); 
    sp.cplex = IloCplex(sp.mod);
    sp.cplex.setWarning(env.getNullStream());
    sp.cplex.setOut(env.getNullStream());
    sp.cplex.setParam(IloCplex::Threads,1);
    sp.cplex.setParam(IloCplex::EpGap, 1.00e-08);          //mipgap
    sp.cplex.setParam(IloCplex::MIPDisplay, 2);            //mipdisplay
    sp.cplex.setParam(IloCplex::NumericalEmphasis,true);   //NumericalEmphasis
    sp.cplex.setParam(IloCplex::EpMrk, 0.01);              //markovitz
    //=========================
    //variables  
    //=========================   
    sp.pi = IloNumVarArray(env,d.m,0.0,+IloInfinity,ILOFLOAT); 
    sp.coefpi = IloNumArray(env,d.m);        
    sp.w = IloNumVarArray(env,d.n,0.0,+IloInfinity,ILOFLOAT);
    sp.lambda = IloNumVarArray(env,d.n,0.0,+IloInfinity,ILOFLOAT);
    sp.g = IloNumVarArray(env,d.n,0.0,1.0,ILOINT);
    sp._g = IloNumArray(env,d.n);
    sp.coefg= IloNumArray(env,d.n);
    sp.constraints = IloRangeArray(env); 
    sp._pi = IloNumArray(env,d.m);

    //===============
    //obj. function- spof
    //===============
        //  //maximize spof : sum{j in J} d[j] * lambda[j] + sum{j in J} d_[j] * w[j] - sum{i in I} _z[i] * pi[i] ;
    IloExpr xpfo(env);
    for(int j = 1; j <= d.n; ++j){   
        xpfo += d.dd[j]*sp.lambda[j-1];
        xpfo += d.dde[j]*sp.w[j-1];      
    }
    for(int i = 1; i <= d.m; ++i){
        xpfo -= sp.pi[i-1];                               
    }
    sp.spof = IloAdd(sp.mod, IloMaximize(env, xpfo));
    xpfo.end();
    //===============
    //constraint
    //===============   
        //  //spr1{i in I, j in J}: lambda[j] - pi[i] <= c[i,j]; 
    for(int i = 1; i <= d.m; ++i){                    
        for(int j = 1; j <= d.n; ++j){
        IloExpr spr1(env);  
        spr1 += sp.lambda[j-1];
        spr1 -= sp.pi[i-1];
        spr1 -= d.c[i][j];
        sp.mod.add(spr1 <= 0.0);
        spr1.end();            
        }
    }
        //  //spr2: sum{j in J} g[j] <= Gamma; 
    IloExpr spr2(env);
    for(int j = 1; j <= d.n; ++j){
        spr2 += sp.g[j-1];
    }
    spr2 -= d.Gamma;
    sp.mod.add(spr2 <= 0.0);
    spr2.end();
        //  //spr3{j in J}: w[j] <= lambda[j];
    for(int j = 1; j <= d.n; ++j){
        IloExpr spr3(env);
        spr3 += sp.w[j-1];
        spr3 -= sp.lambda[j-1];
        sp.mod.add(spr3 <= 0.0);
        spr3.end();                                   
    }
        //  //spr4{j in J}: w[j] <= BigM[j] * g[j];
    for(int j = 1; j <= d.n; ++j){
        IloExpr spr4(env);
        spr4 += sp.w[j-1];
        spr4 -= sp.coefg[j-1]*sp.g[j-1];  
        sp.constraints.add(spr4 <= 0.0);
        spr4.end();                                   
    } 
    sp.mod.add(sp.constraints);
}
// ==============================
//  function solver mp 
// ==============================
void solve_mp(DAT &d, MP_CPX_DAT & mp){ 

    mp.cplex.solve();

    if (mp.IsCCG == 1 && mp.IsBundle == 0){
        mp.lb = mp.cplex.getValue(mp.eta); 
        mp.vars.add(mp.z);
        mp.vars.add(mp.y);
        mp.cplex.getValues(mp._vars,mp.vars);
        for(int i = 0; i < d.m; ++i){
            mp._z[i] = mp._vars[i];
            mp._y[i] = mp._vars[d.m+i];
        }
        //if (mp.iter >= 8 && mp.IsCCG ==1){
            mp.IsCCG = 0;
            mp.IsBundle = 1;
            mp.flag = 0;
                // //solve 
            mp.cplex.setParam(IloCplex::NodeLim, 30);            //Nodes=30
            mp.cplex.setParam(IloCplex::IntSolLim, 1);           //mipsolutions = 4   
            mp.cplex.setParam(IloCplex::TiLim, 5);              //time = 10
                // //printf ("*** Switching to Level and proximal Bundle ***  %10.4f ", mp1.lb);
        //}
    }
    else{
        mp.flag = 1;
        if (mp.cplex.getStatus() == IloAlgorithm::Infeasible || mp.cplex.getStatus() == IloAlgorithm::Unknown  || mp.cplex.getStatus() == IloAlgorithm::Unbounded){
            
            mp.MpInfeasible = 1;            
            mp.IsCCG = 1;
            mp.IsBundle = 0;
                // //coef objective function
            for(int i = 1; i <= d.m; ++i){
                //mp.mpof.setLinearCoef(mp.xsi[i-1], mp.IsBundle);
                mp.mpof.setQuadCoef(mp.e[i-1], mp.e[i-1], mp.IsBundle);
            }
                // //solve 
            mp.cplex.setParam(IloCplex::NodeLim, 9223372036800000000);            //Nodes = default
            mp.cplex.setParam(IloCplex::IntSolLim, 9223372036800000000);          //mipsolutions = default   
            mp.cplex.setParam(IloCplex::TiLim, 1e+75);                            //time = default 
                // //printf("    *** MP Infeasible - Back to CCG ***  %10.4f  ", mp.lb); 
        }
        else{  
            mp.MpInfeasible = 0;
            mp.vars.add(mp.z);
            mp.vars.add(mp.y);
            mp.cplex.getValues(mp._vars,mp.vars);
            for(int i = 0; i < d.m; ++i){
                mp._z[i] = mp._vars[i];
                mp._y[i] = mp._vars[d.m+i];
            }
        }
    }
}
// ==============================
//  function solver spM 
// ==============================
void solve_spM(DAT &d, SP_CPX_DAT &spM, MP_CPX_DAT & mp){
    for(int i = 1; i <= d.m; ++i){
        spM.coefpi[i-1] = -mp._z[i-1]; 
    }                
    spM.spom.setLinearCoefs(spM.pi, spM.coefpi);  
    spM.cplex.solve();
    spM.cplex.getValues(spM.BigM, spM.lambda);

}
// ==============================
//  function solver sp
// ==============================
void solve_sp(DAT &d, SP_CPX_DAT &sp,SP_CPX_DAT &spM, MP_CPX_DAT & mp){
    for(int i = 1; i <= d.m; ++i){
        sp.coefpi[i-1] = -mp._z[i-1]; 
    }        
    sp.spof.setLinearCoefs(sp.pi, sp.coefpi);   
    for(int j = 1; j <= d.n; ++j){
       sp.constraints[j-1].setLinearCoef(sp.g[j-1],-1.0 * spM.BigM[j-1]); 
    }
    sp.cplex.solve();
    sp.cplex.getValues(sp.g,sp._g);
    for (int j = 1; j<=d.n;++j){
        mp.rhsd[j-1] = d.dd[j] + d.dde[j]*sp._g[j-1];
    }   
    
    //============
    // UB
    //============
    sp._spof = (double) sp.cplex.getObjValue();
    if( (mp.IsCCG == 1 && mp.IsBundle == 0)){
        mp.sup = sp._spof;
        for (int i = 1; i <= d.m; ++i){
            mp.sup += 10*d.f[i] * mp._y[i-1];
            mp.sup += d.a[i] * mp._z[i-1];
        }
              
        if (mp.sup < mp.ub){
            mp.ub = mp.sup;
            
            for (int i = 1; i <= d.m; ++i){
                //mp._ze[i-1] = 0.5*mp._ze[i-1] + 0.5*mp._z[i-1];
                mp._ze[i-1] = mp._z[i-1];
            }
                //  //bounds - desvioe constraint
            for(int i = 1; i <= mp._rbsdesvioe.getSize();++i){
                mp._rbsdesvioe[i-1] = -1.0*mp._ze[i-1];
                mp._lbsdesvioe[i-1] = -1.0*mp._ze[i-1];
            }
            mp.desvioe.setBounds(mp._lbsdesvioe, mp._rbsdesvioe);
        }
    }
    else{
        mp.sup = sp._spof;
        for (int i = 1; i <= d.m; ++i){
            mp.sup += 10*d.f[i] * mp._y[i-1];
            mp.sup += d.a[i] * mp._z[i-1];
        }
        // //inicialization parameter tk
        //===================================
        if(mp.iter == 8){            
            sp.cplex.getValues(sp.pi,sp._pi);
            double gradf;
            for(int i = 1; i <= d.m; ++i){
                if(mp._z[i - 1] != 0.00){
                    gradf -= sp._pi[i - 1];
                    gradf += d.a[i];
                }
                    gradf += 10*d.f[i]*mp._y[i-1];
            }
            mp._tk = abs(gradf);
        }
        // //end inicialization parameter tk
        //===================================

        if (mp.sup < mp.ub ){
            mp.ub = mp.sup;
            for (int i = 1; i <= d.m; ++i){
                //mp._ze[i-1] = 0.5*mp._ze[i-1] + 0.5*mp._z[i-1];
                mp._ze[i-1] = mp._z[i-1];
            }
                //  //bounds - desvioe constraint
            for(int i = 1; i <= mp._rbsdesvioe.getSize(); ++i){
                mp._rbsdesvioe[i-1] = -1.0*mp._ze[i-1];
                mp._lbsdesvioe[i-1] = -1.0*mp._ze[i-1];
            }
            mp.desvioe.setBounds(mp._lbsdesvioe, mp._rbsdesvioe);
            
            // //calculation parameter tk
            //===================================
            mp._tk0 = mp._tk;
            if(mp._count == 0){
                mp._tk = 2*mp._tk0;    
            }
            else{
                mp._tk = mp._tk0;
            }
            mp._count = 0;           

            //===================================
                //  //coef objective function
            double coefx = 1/(2*mp._tk);
            for(int i = 1; i <= d.m; ++i){
                //mp.mpof.setLinearCoef(mp.xsi[i-1], coefx*mp.IsBundle);
                mp.mpof.setQuadCoef(mp.e[i-1], mp.e[i-1], coefx*mp.IsBundle);
            }
        } 
        else{
            
            if(mp.flag == 1 && mp._count > 0)
            {
                    //  //printf("*** Back to CCG *** ");
                mp.IsCCG = 1;
                mp.IsBundle = 0;               
                    //  //solve         
                mp.cplex.setParam(IloCplex::NodeLim, 9223372036800000000);            //Nodes = default
                mp.cplex.setParam(IloCplex::IntSolLim, 9223372036800000000);          //mipsolutions = default   
                mp.cplex.setParam(IloCplex::TiLim, 1e+75);                            //time = default 
            }
            // //calculation parameter tk
            //===================================
            mp._tk0 = mp._tk;
            if(mp._count > 3){
                mp._tk = 2*mp._tk0; 
                mp._count = 1;
            }else{
                mp._tk = mp._tk0; 
                int count = mp._count;
                mp._count = max(count+1,1);
            }
            //===================================           
            //  //coef objective function
            double coefx = 1/(2*mp._tk);
            for(int i = 1; i <= d.m; ++i){
                //mp.mpof.setLinearCoef(mp.xsi[i-1], coefx*mp.IsBundle);
                mp.mpof.setQuadCoef(mp.e[i-1], mp.e[i-1], coefx*mp.IsBundle);
            }
        }
    }
}
// ==============================
//  solution
// ==============================
void Print_solution(DAT &d,MP_CPX_DAT &mp, SP_CPX_DAT &sp){
        int h = mp.iter;
        double Tp = mp.Timeprob;
        double Gp = 100*mp.gap;
        int nb = mp.nobb;
        double TempoporsegMaster = mp.TimeMaster/h;

//===========
// solution 
//===========
        FILE *arquivo1;
        arquivo1=fopen("QP-PBCCG-Result-Vanilla.txt","aw+");
        if(arquivo1 == NULL){
            printf("Não foi possivel abrir o arquivo1");
            exit(0);
        }
        fprintf (arquivo1,"%2s %2s %1s %5d %2s %1s %5d %2s %1s %10.2f %2s %1s %10.2f %2s %1s %8.2f %2s %1s %8.2f %2s %1s %8.2f %2s %1s  %8.2f\n",d.name,"&","&", h,"&","&", nb,"&","&", mp.lb,"&","&", mp.ub,"&","&", Gp,"&","&", mp.TimeMaster,"&" ,"&" ,TempoporsegMaster, "&", "&" ,Tp);

        fclose(arquivo1);
}
