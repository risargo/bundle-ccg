//===================================================
// Location transportation problem
// Authors: Débora Ribeiro - deboralvesribeiro@gmail.com 
//          Ricardo Camargo - risargo@gmail.com
//          Gilberto Miranda - gilbertomirandajr@gmail.com
// C&CG- Column and constraint generation 
//===================================================
#include <iostream> 
#include <iomanip>
#include <fstream>
#include <vector>
#include <cmath>
using namespace std;
#include <ilcplex/ilocplex.h>
#define EPSILON 1e-08
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
    double Gamma;                  // uncertainty budget 
    double ex;
    char name[250];
} DAT;
// ==============================================
// struct of mp and sp
// ==============================================
typedef struct{ 
    IloNum sup;
    IloNum ub;
    IloNum lb;
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
    IloNumVar eta;
    IloNumArray _z;
    IloNumArray _y;
    IloObjective mpof;
    IloRangeArray constraints; 
    IloRangeArray constraintssystems1;
    IloRangeArray constraintssystems2;
    IloRangeArray constraintssystems3;
    IloNumArray rhsd; 
    IloNumVarArray vars;
    IloNumArray _vars;
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
    IloNumArray _g;
    IloRangeArray constraints;
    IloRangeArray constraintsBigM;
    IloObjective spof;
    IloObjective spom;
    IloNumArray BigM;
} SP_CPX_DAT;

void help();
void read_data(char name[],DAT &d);                                       
void create_model_mp(DAT &d, MP_CPX_DAT &mp);
void create_model_spM(DAT &d, SP_CPX_DAT &spM);
void create_model_sp(DAT &d, SP_CPX_DAT &sp, SP_CPX_DAT &spM);
void solve_mp(DAT &d, MP_CPX_DAT &mp);
void solve_spM(DAT &d, SP_CPX_DAT &spM, MP_CPX_DAT &mp);
void solve_sp(DAT &d, SP_CPX_DAT &sp,SP_CPX_DAT &spM, MP_CPX_DAT &mp);
void add_constraintssystems(DAT &d, MP_CPX_DAT &mp);
void iniciatilization(DAT &d, MP_CPX_DAT & mp);
void Print_solution(DAT &d,MP_CPX_DAT &mp, SP_CPX_DAT &sp);
// ============================================================================================

// main program
// ============================================================================================
int main ( int argc,  char *argv[]){
    // ===============
    // dat
    // ===============
    DAT d;
    if (argc < 2){
        help();
    }
    d.ex = (argc > 2) ? atof(argv[2]) : 1.0;
    read_data(argv[1],d);
    int nh = 0;  int nnobb = 0; double TimeMasterInicial = 0;
    // ===============
    // cplex environment
    // ===============
    MP_CPX_DAT mp;
    SP_CPX_DAT spM;
    SP_CPX_DAT sp;
    
    try {
            //  //printf("\nLocation transportation model \n\n");
        IloTimer crono(mp.env);
        mp.crono = &crono;
        
        create_model_mp(d, mp);
        create_model_spM(d, spM);
        create_model_sp(d, sp, spM);
        // ===============
        // main loop
        // ===============
        iniciatilization(d, mp);
        //solve_spM(d, spM, mp);
        crono.start();
        while(mp.gap > EPSILON and crono.getTime() <= 21600 ){
            // ===========
            // mp
            // ===========
            TimeMasterInicial = crono.getTime();
            solve_mp(d, mp);           
            mp.TimeMaster = mp.TimeMaster + (crono.getTime() - TimeMasterInicial);
            nnobb = nnobb + mp.cplex.getNnodes();
            // ===========
            // sp
            // ===========
            solve_spM(d, spM, mp);
            solve_sp(d, sp, spM, mp);
            
            mp.ub = mp.ub > mp.sup ? mp.sup : mp.ub;
            ++mp.it;
            add_constraintssystems(d, mp);
            
            mp.gap = (mp.ub - mp.lb)/mp.ub;
                ++nh;
                printf ("%5d |%5d | %8.4f | %8.4f | %8.4f | %8.4f\n",nh , nnobb, mp.lb, mp.ub, 100*mp.gap, crono.getTime());
        }
        mp.Timeprob = crono.getTime();
        mp.nobb = nnobb;

        Print_solution(d,mp,sp);
    }
    catch (IloException& ex) {                                   
        cerr << "Error: " << ex << endl;
    }
    return 0;
}
// ============================================================================================
// ============================================================================================

// ==============================
// function help
// ==============================
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
    register int i;
    register int j;
        // // demand d e d_
    for (j = 1; j <= d.n; ++j){
        arq >> d.dd[j];
        arq >> d.dde[j];
    }
        // // fixed costs, capacity costs e capacity of the facility
    for (i = 1; i <= d.m; ++i){
        arq >> d.f[i];
        arq >> d.a[i];
        arq >> d.k[i];
    }   
        // // transportation costs;
    for (i = 1; i <= d.m; ++i){
        for ( j = 1; j <= d.n; ++j){
            arq >> d.c[i][j];
        }
    }
    arq.close();
}
// ==============================
// iniciatilization
// ==============================
void iniciatilization(DAT &d, MP_CPX_DAT & mp){
    mp.ub = IloInfinity;     mp.lb = 0;     mp.it = 0;    mp.gap = 1; 
}
// ==============================
//  constrains  - mp 
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
        //  //mpcut{h in H}: eta >= sum{i in I, j in J} c[i,j] * x[h,i,j];
    IloExpr r31mpcut(env);
    r31mpcut += mp.eta;  
    mp.constraintssystems3.add(r31mpcut >= 0);
    r31mpcut.end();

    mp.mod.add(mp.constraintssystems1); 
    mp.mod.add(mp.constraintssystems2); 
    mp.mod.add(mp.constraintssystems3); 
    
    // ===============   
    // function Add x
    // ===============   
    for ( i = 1; i <= d.m; ++i){
        for ( j = 1; j <= d.n; ++j){
            IloNumColumn newCol = mp.mpof (0.00);
            newCol += mp.constraintssystems1[(mp.it-1)*(d.n) + i-1](1.00);            
            newCol += mp.constraintssystems2[(mp.it-1)*d.m + j-1](1.00);
            newCol += mp.constraintssystems3[mp.it-1](-1.00*d.c[i][j]);
            IloNumVar newVar(newCol, 0.0, +IloInfinity,ILOFLOAT);
            mp.mod.add(newVar);
            mp.x.add(newVar);
        }
    }
}
// ==============================
// Function mp problem
// ==============================
void create_model_mp(DAT &d, MP_CPX_DAT &mp){
    IloEnv env = mp.env;
    mp.mod = IloModel(env); 
    mp.cplex = IloCplex(mp.mod);  
    mp.cplex.setWarning(env.getNullStream());
    mp.cplex.setOut(env.getNullStream());
    mp.cplex.setParam(IloCplex::Threads,1);
    mp.cplex.setParam(IloCplex::EpGap, 1.00e-08);       //mipgap
    mp.cplex.setParam(IloCplex::MIPDisplay, 2);         //mipdisplay
    mp.cplex.setParam(IloCplex::NumericalEmphasis,1);   //NumericalEmphasis
    mp.cplex.setParam(IloCplex::EpMrk, 0.01); 
    //=========================
    //variables  
    //=========================   
    mp.x = IloNumVarArray(env);
    mp.y = IloNumVarArray(env,d.m,0.0,1.0,ILOINT);
    mp._y = IloNumArray(env,d.m);
    mp.z = IloNumVarArray(env,d.m,0.0,+IloInfinity,ILOFLOAT);
    mp._z = IloNumArray(env,d.m);
    mp.rhsd = IloNumArray(env,d.n);
    mp.eta = IloNumVar(env,0.0,+IloInfinity,ILOFLOAT);
    mp.constraints = IloRangeArray(env);
    mp.constraintssystems1 = IloRangeArray(env);
    mp.constraintssystems2 = IloRangeArray(env);
    mp.constraintssystems3 = IloRangeArray(env);
    mp.vars = IloNumVarArray(env);
    mp._vars = IloNumArray(env);
    
    register int i;
    register int j;
    mp.maxD = 0.0 ;    
    for( j = 1; j <= d.n; ++j){
        mp.maxD += d.dd[j]+d.dde[j];
    }
//===============
//obj. function- mpof
//===============
        //  //minimize mpof: sum{i in I} 10*f[i] * y[i] + sum{i in I} a[i] * z[i] + eta;
    IloExpr xpfo(env);
    xpfo += mp.eta;
    for( i = 1; i <= d.m; ++i){
        xpfo += 10*d.f[i] * mp.y[i-1] + d.a[i] * mp.z[i-1];
    }
    mp.mpof = IloAdd(mp.mod, IloMinimize(env, xpfo));
    xpfo.end();     
    //===============
    //constrain
    //===============   
        //  //mpfeasibility: sum{i in I} z[i] >=  _maxD;    
    IloExpr r1mpfeasibility(env);
    for( i = 1; i <= d.m; ++i){
        r1mpfeasibility += mp.z[i-1];       
    }
        r1mpfeasibility -= mp.maxD;                      
        mp.constraints.add(r1mpfeasibility >= 0.0);
        r1mpfeasibility.end();                                   
        
        //  //mpcapacity{i in I}: z[i] <= 6*k[i] * y[i];   
    for( i = 1; i <= d.m; ++i){
        IloExpr r2mpcapacity(env);
        r2mpcapacity += mp.z[i-1];               
        r2mpcapacity -= 6*d.k[i]*mp.y[i-1];
        mp.constraints.add(r2mpcapacity <= 0.0);
        r2mpcapacity.end();                                   
    }
        //  //mpcut{h in H}: eta >= sum{i in I, j in J} c[i,j] * x[h,i,j];
    IloExpr r31mpcut(env);
    r31mpcut += mp.eta;   
    mp.constraintssystems3.add(r31mpcut >= 0);
    r31mpcut.end();
    
    mp.mod.add(mp.constraints);
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
    
    register int i;
    register int j;
    //===============
    //obj. function- spoM
    //===============
        //  //maximize spoM: sum{j in J}( ( d[j] + d_[j] ) * lambda[j] ) - sum{i in I}( _z[i] * pi[i] );
    IloExpr xpfo(env);
    for( j = 1; j <= d.n; ++j){   
        xpfo += (d.dd[j]+d.dde[j])*spM.lambda[j-1];
    }
    for( i = 1; i <= d.m; ++i){
        xpfo -= spM.pi[i-1]; 
    }
    spM.spom = IloAdd(spM.mod, IloMaximize(env, xpfo));
    xpfo.end();
    //===============
    //constrain
    //===============   
        //  //spr1{i in I, j in J}: lambda[j] - pi[i] <= c[i,j]; 
    for( i = 1;i <= d.m; ++i){
        for( j = 1; j <= d.n; ++j){
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
    sp.cplex.setParam(IloCplex::EpGap, 1.00e-08);       //mipgap
    sp.cplex.setParam(IloCplex::MIPDisplay, 2);         //mipdisplay
    sp.cplex.setParam(IloCplex::NumericalEmphasis,1);   //NumericalEmphasis
    sp.cplex.setParam(IloCplex::EpMrk, 0.01); 
    //=========================
    //variables  
    //=========================   
    sp.pi = IloNumVarArray(env,d.m,0.0,+IloInfinity,ILOFLOAT); 
    sp.coefpi = IloNumArray(env,d.m);     
    sp.w = IloNumVarArray(env,d.n,0.0,+IloInfinity,ILOFLOAT);
    sp.lambda = IloNumVarArray(env,d.n,0.0,+IloInfinity,ILOFLOAT);
    sp.g = IloNumVarArray(env,d.n,0.0,1.0,ILOINT);
    sp._g = IloNumArray(env,d.n);
    sp.constraints = IloRangeArray(env); 
    sp.constraintsBigM = IloRangeArray(env);    
    
    register int i;
    register int j;
    //===============
    //obj. function- spof
    //===============
        //  //maximize spof : sum{j in J} d[j] * lambda[j] + sum{j in J} d_[j] * w[j] - sum{i in I} _z[i] * pi[i] ;
    IloExpr xpfo(env);
    for( j = 1; j <= d.n; ++j){   
        xpfo += d.dd[j]*sp.lambda[j-1] + d.dde[j]*sp.w[j-1];;
    }
    for( i = 1; i <= d.m; ++i){
        xpfo -= sp.pi[i-1];                               
    }
    sp.spof = IloAdd(sp.mod, IloMaximize(env, xpfo));
    xpfo.end();
    //===============
    //constrain
    //===============   
        //  //spr1{i in I, j in J}: lambda[j] - pi[i] <= c[i,j]; 
    for( i = 1; i <= d.m; ++i){                    
        for( j = 1; j <= d.n; ++j){
            IloExpr spr1(env);  
            spr1 += sp.lambda[j-1];
            spr1 -= sp.pi[i-1];
            spr1 -= d.c[i][j];
            sp.constraints.add(spr1 <= 0.0);
            spr1.end();            
        }
    }
        //  //spr2: sum{j in J} g[j] <= Gamma; 
    IloExpr spr2(env);
    for( j = 1; j <= d.n; ++j){
        spr2 += sp.g[j-1];
    }
    spr2 -= d.Gamma;
    sp.constraints.add(spr2 <= 0.0);
    spr2.end();
        //  //spr3{j in J}: w[j] <= lambda[j];
    for( j = 1; j <= d.n; ++j){
        IloExpr spr3(env);
        spr3 += sp.w[j-1];
        spr3 -= sp.lambda[j-1];
        sp.constraints.add(spr3 <= 0.0);
        spr3.end();                                   
    }
        //  //spr4{j in J}: w[j] <= BigM[j] * g[j];
    for( j = 1; j <= d.n; ++j){
        IloExpr spr4(env);
        spr4 += sp.w[j-1];
        spr4 -= sp.g[j-1]; 
        sp.constraintsBigM.add(spr4 <= 0.0);
        spr4.end();                                   
    } 
    sp.mod.add(sp.constraints);
    sp.mod.add(sp.constraintsBigM);
}
// ==============================
//  function solver mp 
// ==============================
void solve_mp(DAT &d, MP_CPX_DAT &mp){
    mp.cplex.solve();                               
    mp.lb=(double) mp.cplex.getObjValue();          
    mp.vars.add(mp.z);
    mp.vars.add(mp.y);
    mp.cplex.getValues(mp._vars,mp.vars);
    for(int i = 0; i < d.m; ++i){
        mp._z[i] = mp._vars[i];
        mp._y[i] = mp._vars[d.m+i];
    }
}
// ==============================
//  function solver spM 
// ==============================
void solve_spM(DAT &d, SP_CPX_DAT &spM, MP_CPX_DAT &mp){
    for(int i = 1; i <= d.m; ++i){
        spM.coefpi[i-1] = -1.0*mp._z[i-1]; 
    }
    spM.spom.setLinearCoefs(spM.pi, spM.coefpi);         
    spM.cplex.solve();
    spM.cplex.getValues(spM.BigM, spM.lambda );
}
// ==============================
//  function solver sp
// ==============================
void solve_sp(DAT &d, SP_CPX_DAT &sp,SP_CPX_DAT &spM, MP_CPX_DAT &mp){
    register int i;
    register int j;
    for( i = 1; i <= d.m; ++i){                     
        sp.coefpi[i-1] =  -1.0*mp._z[i-1];               
    } 
    sp.spof.setLinearCoefs(sp.pi, sp.coefpi);  
    for( j = 1; j <= d.n; ++j){
       sp.constraintsBigM[j-1].setLinearCoef(sp.g[j-1],-1.0 * spM.BigM[j-1]); 
    }   
    sp.cplex.solve();
    sp.cplex.getValues(sp.g,sp._g);
    for ( j = 1; j<=d.n;++j){
        mp.rhsd[j-1] = d.dd[j] + d.dde[j]*sp._g[j-1];
    }    
        //  //for ub
    mp.sup = (double) sp.cplex.getObjValue();
    for ( i = 1; i <= d.m; ++i){
        mp.sup += 10*d.f[i] * mp._y[i-1] + d.a[i] * mp._z[i-1];;
    }
}
// ==============================
//  solution
// ==============================
void Print_solution(DAT &d,MP_CPX_DAT &mp, SP_CPX_DAT &sp){
        int h = mp.it;
        double Tp = mp.Timeprob;
        double Gp = 100*mp.gap;
        int nb = mp.nobb;
        double TempoporsegMaster = mp.TimeMaster/h;
        //===========              
        // solution 
        //===========   
        FILE *arquivo1;
        arquivo1=fopen("CCG-Result.txt","aw+");
        if(arquivo1 == NULL){
            printf("Não foi possivel abrir o arquivo1");
            exit(0);
        }
        fprintf (arquivo1,"%2s %2s %1s %5d %2s %1s %5d %2s %1s %10.2f %2s %1s %10.2f %2s %1s %8.2f %2s %1s %8.2f %2s %1s %8.2f %2s %1s \n",d.name,"&","&", h,"&","&", nb,"&","&", mp.lb,"&","&", mp.ub,"&","&", Gp,"&","&", TempoporsegMaster ,"&" ,"&", Tp,"&" ,"&");
        fclose(arquivo1);

}

