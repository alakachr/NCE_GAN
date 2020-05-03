#include <iostream>
using namepace std;

int main() { 
    int a, b;

    cout << " Choisissez une valeur pour a"<< endl;
    cin >> a;
    
    
    ajouter_copie(a,b);
    cout << " après exp. copie la valeur de b est " << b;

    b =0;
    ajouter_ref(a,b);
    cout << " après exp. copie la valeur de b est " << b;

    b = 0;
    ajouter_pointeur(&a,&b);
    cout << " après exp. copie la valeur de b est " << b;


}
///////////// Exercice 1: ( 3 manières) 


int ajouter_copie(int p, int a){
    a = p;

    return a;
}


int ajouter_ref(int &p,int &a){
    a=p;
}

int ajouter_pointeur( int *p, int *a){
    *a=*p;
}