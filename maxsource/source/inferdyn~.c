/**
	@file
	inferdyn~: inference from onnx model.
	original by: shaltiel eloul
	@ingroup cuda
*/
#define FFTW

#include "ext.h"
#include "ext_obex.h"
#include "z_dsp.h"
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include<dlfcn.h>
#ifdef FFTW
#include </usr/local/include/fftw3.h>
#endif
// struct to represent the object's state
typedef struct _inferdyn {
	t_pxobject		ob;			// the object itself (t_pxobject in MSP instead of t_object)
	int		offs; 	// the value of a property of our object
    bool ready_record;
    int nlabels;
    float preseqsum;
    
    float * (* inference) (float *,float *,int, long);
    float * (* test) (float *);
    const char *  (* initial) (const char *);
    float * inference_array;
    float * valuation_array;
    int mode;
    int is_infer;
    int samplesize;
    int fftsize;
    int filesize;
    int repeats;
    int rep;
    complex double * filter_L;
    complex double * filter_R;

    int loading_flag;
    long loading_no;
    int * label_vector;
    
    complex double * dft_freq;
    
    double * time2;
    float threshold;

    void *out;
    

} t_inferdyn;


// method prototypes
void loadNN(t_inferdyn *x,t_symbol *s, long argc,t_atom *argv);
void loadDLpaths(t_inferdyn *x,t_symbol *s, long argc,t_atom *argv);


void record_training(t_inferdyn *x, int sampleframes);
void inference_sample(t_inferdyn *x, int halframes,float norm);

void *inferdyn_new(t_symbol *s, long argc, t_atom *argv);
void inferdyn_free(t_inferdyn *x);
void threshold(t_inferdyn *x, float);
void inferdyn_assist(t_inferdyn *x, void *b, long m, long a, char *s);
void inferdyn_anything(t_inferdyn *x, t_symbol *s, long ac, t_atom *av);
void inferdyn_float(t_inferdyn *x, double f);
void infer_event(t_inferdyn *x);
void inferdyn_dsp64(t_inferdyn *x, t_object *dsp64, short *count, double samplerate, long maxvectorsize, long flags);
void inferdyn_perform64(t_inferdyn *x, t_object *dsp64, double **ins, long numins, double **outs, long numouts, long sampleframes, long flags, void *userparam);


// global class pointer variable
static t_class *inferdyn_class = NULL;


//***********************************************************************************************

void ext_main(void *r)
{
	// object initialization, note the use of dsp_free for the freemethod, which is required
	// unless you need to free allocated memory, in which case you should call dsp_free from
	// your custom free function.

	t_class *c = class_new("inferdyn~", (method)inferdyn_new, (method)dsp_free, (long)sizeof(t_inferdyn), 0L, A_GIMME, 0);
    class_addmethod(c, (method)inferdyn_free,	"close_files",	A_GIMME, 0);
	class_addmethod(c, (method)inferdyn_float,		"float",	A_FLOAT, 0);
    class_addmethod(c, (method)threshold,		"threshold",	A_FLOAT, 0);
    class_addmethod(c, (method)infer_event, "bang", 0);
	class_addmethod(c, (method)inferdyn_dsp64,		"dsp64",	A_CANT, 0);
	class_addmethod(c, (method)inferdyn_assist,	"assist",	A_CANT, 0);
    class_addmethod(c, (method)inferdyn_anything,		"list",	A_GIMME, 0);
    class_addmethod(c, (method)loadNN, "LoadNN", A_GIMME, 0);
    class_addmethod(c, (method)loadDLpaths, "LoadDLPaths", A_GIMME, 0);
    // read-only
    class_addmethod(c, (method)inferdyn_assist,			"assist",		A_CANT, 0);
	class_dspinit(c);
	class_register(CLASS_BOX, c);
	inferdyn_class = c;
}


void loadNN(t_inferdyn *x,t_symbol *s, long argc,t_atom *argv)
{
    
    const char * meta= x->initial(atom_getsym(argv)->s_name);
    char * d;
    post("%s",meta);
    x->loading_no = strtol(meta, &d, 10);
    post("%d",x->loading_no);
    x->loading_flag+=1;
}

void loadDLpaths(t_inferdyn *x,t_symbol *s, long argc,t_atom *argv)
{
    
    if (argc<3)
    {
        post("Three DLLs needed: libonnxruntime.1.1.0.dylib, libonnxruntime.dylib, and the NN model");
        return;
    }
    void *handle;
    char *error;

    handle = dlopen (atom_getsym(argv)->s_name, RTLD_LAZY);
    if (!handle) {
        post("%s",dlerror());
        post("%s","error");
    }
    handle = dlopen (atom_getsym(argv+1)->s_name, RTLD_LAZY);
    if (!handle) {
        post("%s",dlerror());
        post("%s","error");
    }
    handle = dlopen (atom_getsym(argv+2)->s_name, RTLD_LAZY);
    if (!handle) {
        post("%s",dlerror());
        post("%s","error");
    }

    x->inference = (float*(*) (float *,float *,int,long)) dlsym(handle, "inference");
    if ((error = dlerror()) != NULL)  {
        post("%s",error);
        return;
    }
    x->test = (float*(*) (float *)) dlsym(handle, "test");
    if ((error = dlerror()) != NULL)  {
        post("%s",error);
        return;
    }
    
    x->initial = (const char *(*) (const char *)) dlsym(handle, "initial");
    if ((error = dlerror()) != NULL)  {
        post("%s",error);
        return;
    }
    x->loading_flag=+1;
    post("%s","DLLs loaded");
}


void *inferdyn_new(t_symbol *s, long argc, t_atom *argv)
{
    t_inferdyn *x = (t_inferdyn *)object_alloc(inferdyn_class);
    x->samplesize=1024;
    x->nlabels=88;
    x->threshold=0.4;
    x->mode=0;
    #ifdef FFTW
    x->mode=1;
    #endif
    if (argc>0)
        x->samplesize = atom_getlong(argv);
    if (argc>1)
         x->repeats = atom_getlong(argv+1);
    if (argc>2)
        x->nlabels=atom_getlong(argv+2);
    if (argc>3)
        x->mode=atom_getlong(argv+3);
    
    x->is_infer=false;

    int dspsize=x->samplesize*2;
    x->rep=0;
    x->dft_freq = (complex double*)malloc(sizeof(complex double)*(dspsize+1));
    x->inference_array= (float *)malloc((x->repeats*x->samplesize)*sizeof(float));
    
    x->valuation_array= (float *)malloc((x->nlabels)*sizeof(float));
    
    x->time2 = malloc(dspsize*2*sizeof(double));
    
    x->label_vector = malloc(x->nlabels*sizeof(double));
    x->ready_record=false;
    x->offs=0;
    x->loading_flag=0;
	if (x) {
		dsp_setup((t_pxobject *)x, 1);
        x->out = outlet_new(x, "NULL");

		outlet_new(x, "signal");
       	}
	return (x);
  
}

// NOT CALLED!, we use dsp_free for a generic free function
void inferdyn_free(t_inferdyn *x)
{

}


void inferdyn_assist(t_inferdyn *x, void *b, long m, long a, char *s)
{
	if (m == ASSIST_INLET) { //inlet
		sprintf(s, "I am inlet %ld", a);
	}
	else {	// outlet
		sprintf(s, "I am outlet %ld", a);
	}
}


void inferdyn_float(t_inferdyn *x, double f)
{
    x->offs=0;
}


// registers a function for the signal chain in Max
void inferdyn_dsp64(t_inferdyn *x, t_object *dsp64, short *count, double samplerate, long maxvectorsize, long flags)
{
	object_method(dsp64, gensym("dsp_add64"), x, inferdyn_perform64, 0, NULL);
}

// this is the 64-bit perform method audio vectors
void inferdyn_perform64(t_inferdyn *x, t_object *dsp64, double **ins, long numins, double **outs, long numouts, long sampleframes, long flags, void *userparam)
{
	t_double *inL = ins[0];
    t_double *outL = outs[0];
    
    if (x->loading_flag<2)
    {
        return;
    }
    int n = sampleframes;
    float norm=0;
    while(n--)
    {
        // noramlising
        if (n<x->samplesize) norm+= inL[n]*inL[n];
        x->time2[n] =inL[n];
        x->time2[sampleframes + n] = 0; // for fft buffer in fftw
    }
    norm=sqrtf(norm);
    float squsum=norm;
    n = sampleframes;
    
    #ifdef FFTW
    if (x->mode==1) //calculate fft from fftw.
    {
        norm=0;
        n = sampleframes;
        fftw_plan fft = fftw_plan_dft_r2c_1d(sampleframes, x->time2, x->dft_freq, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
        fftw_execute(fft);
        for(int i=0;i<x->samplesize;i++)
        {
            x->time2[i]=creal(x->dft_freq[i]);
            outL[i]=x->time2[i];
            norm+=outL[i]*outL[i];
        }
        norm = sqrtf(norm);
    }
    #endif //FFTW
   
    
    if ((x->is_infer && squsum > x->threshold) || x->rep>0)
    {
        inference_sample(x, x->samplesize, norm); //frequency domain inference
    }

}

void inference_sample(t_inferdyn *x, int nsamples, float norm)
{
    if(x->rep<x->repeats)
    {
        for(int i=0;i<nsamples;i++)
        {
            x->inference_array[i+x->rep*nsamples]=fabs(x->time2[i])/norm;
        }
        x->rep++;
    }
//    post("rep %d",x->rep);
    
    if (x->rep == x->repeats)
    {
        x->inference(x->inference_array,x->valuation_array,x->repeats*nsamples,x->loading_no);
        
        int maxi = 0;
        double maxv = -100.0;
        for (int i=0; i<x->nlabels;i++)
        {
            float temp = x->valuation_array[i];
            maxv = maxv > temp ? maxv:temp;
            maxi = maxv > temp ? maxi:i;

        }
        outlet_int(x->out, maxi);
//        post("%d\t %f",maxi,maxv);
        x->ready_record=false;
        x->is_infer=false;
        x->rep=0;
    }
}


void infer_event(t_inferdyn *x)
{
    if (x->loading_flag<2)
    {
        post("needs to load DLLs and ONNX using: LoadDLPaths & LoadNN messages");
    }
    x->is_infer=true;
    
}

void inferdyn_anything(t_inferdyn *x, t_symbol *s, long ac, t_atom *av)
{

}

void threshold(t_inferdyn *x, float f)
{
    x->threshold = f;
    post("new threshold set");
}





