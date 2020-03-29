/**
	@file
	traindyn~: a simple training set generator from signal/freq, object for Max/MSP
	original by: shaltiel eloul
	@ingroup ML
*/
//#define FFTW //if wish to use fftw to do fft to signal.

#include "ext.h"			// standard Max include, always required (except in Jitter)
#include "ext_obex.h"		// required for "new" style objects
#include "z_dsp.h"			// required for MSP objects
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include<dlfcn.h>

#ifdef FFTW
#include </usr/local/include/fftw3.h>
#endif
// struct to represent the object's state
typedef struct _traindyn {
	t_pxobject		ob;			// the object itself (t_pxobject in MSP instead of t_object)
    bool trains;
    int n_trains;
    int samplesize;
    bool ready_record;
    int nlabels;
    float preseqsum;
    int mode;
    int parts;
    int fftsize;
    int filesize;
    int repeats;
    int rep;
    bool label_assigned;
    bool flag_loadpath;
    FILE * f_trains;
    FILE * f_labels;
    char * path_trains;
    char * path_labels;
    
    int * label_vector;
    
    complex double * dft_freq;
    
    double * time2;
    float threshold;

    void *out;
    

} t_traindyn;


// method prototypes
void loadpaths(t_traindyn *x,t_symbol *s, long argc,t_atom *argv);
void record_training(t_traindyn *x, int sampleframes);
void new_signal(t_traindyn *x);

void *traindyn_new(t_symbol *s, long argc, t_atom *argv);
void traindyn_free(t_traindyn *x);
void threshold(t_traindyn *x, float);
void traindyn_assist(t_traindyn *x, void *b, long m, long a, char *s);
void traindyn_anything(t_traindyn *x, t_symbol *s, long ac, t_atom *av);
void new_label(t_traindyn *x, t_symbol *s, long ac, t_atom *av);
void traindyn_float(t_traindyn *x, double f);
void traindyn_dsp64(t_traindyn *x, t_object *dsp64, short *count, double samplerate, long maxvectorsize, long flags);
void traindyn_perform64(t_traindyn *x, t_object *dsp64, double **ins, long numins, double **outs, long numouts, long sampleframes, long flags, void *userparam);




// global class pointer variable
static t_class *traindyn_class = NULL;


//***********************************************************************************************

void ext_main(void *r)
{
	// object initialization, note the use of dsp_free for the freemethod, which is required
	// unless you need to free allocated memory, in which case you should call dsp_free from
	// your custom free function.

	t_class *c = class_new("traindyn~", (method)traindyn_new, (method)dsp_free, (long)sizeof(t_traindyn), 0L, A_GIMME, 0);
    class_addmethod(c, (method)traindyn_free,	"close_files",	A_GIMME, 0);
	class_addmethod(c, (method)traindyn_float,		"float",	A_FLOAT, 0);
    class_addmethod(c, (method)threshold,		"threshold",	A_FLOAT, 0);
	class_addmethod(c, (method)traindyn_dsp64,		"dsp64",	A_CANT, 0);
	class_addmethod(c, (method)traindyn_assist,	"assist",	A_CANT, 0);
    class_addmethod(c, (method)traindyn_anything,		"list",	A_GIMME, 0);
    class_addmethod(c, (method)new_label,		"new_label",	A_GIMME, 0);
    class_addmethod(c, (method)new_signal,	"new_signal",	A_GIMME, 0);
    class_addmethod(c, (method)loadpaths, "LoadPaths", A_GIMME, 0);

    // read-only
    class_addmethod(c, (method)traindyn_assist,			"assist",		A_CANT, 0);
    CLASS_ATTR_LONG(c, "parts", 0 /*ATTR_SET_OPAQUE_USER*/, t_traindyn, parts);

	class_dspinit(c);
	class_register(CLASS_BOX, c);
	traindyn_class = c;
}


void loadpaths(t_traindyn *x,t_symbol *s, long argc,t_atom *argv)
{
    if ((argv)->a_type == A_SYM) {
        char* s1 = "trains";
        char* s2 = "labels";
        x->path_trains = (char*)malloc((strlen(atom_getsym(argv)->s_name)+1+strlen(s1)) * sizeof(char));
        x->path_labels= (char*)malloc((strlen(atom_getsym(argv)->s_name)+1+ strlen(s2)) * sizeof(char));
        strcpy(x->path_trains, atom_getsym(argv)->s_name);
        strcat(x->path_trains, s1);
        strcpy(x->path_labels, atom_getsym(argv)->s_name);
        strcat(x->path_labels, s2);
    }
    else
    {
        post("something worng with paths");
    }
    
    // open the file for writing
    x->f_trains = fopen (x->path_trains,"a");
    x->f_labels = fopen (x->path_labels,"a");
    x->flag_loadpath=true;
}


void *traindyn_new(t_symbol *s, long argc, t_atom *argv)
{
    
    t_traindyn *x = (t_traindyn *)object_alloc(traindyn_class);
    x->trains=true;
    x->samplesize=2048;
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
    
    x->rep=x->repeats;
    x->label_assigned=true;
    int dspsize=x->samplesize*2;
    x->dft_freq = (complex double*)malloc(sizeof(complex double)*(dspsize+1));
    x->time2 = malloc(dspsize*sizeof(double));
    x->label_vector = malloc(x->nlabels*sizeof(double));
    x->n_trains=0;
    x->ready_record=false;
    x->flag_loadpath=false;
	if (x) {
		dsp_setup((t_pxobject *)x, 1);
        x->out = outlet_new(x, "NULL");

		outlet_new(x, "signal");
        outlet_new(x, "signal");
       	}

	return (x);
    
}

void traindyn_free(t_traindyn *x)
{
    fclose(x->f_trains);
    fclose(x->f_labels);
    dsp_free(&x->ob);
}


void traindyn_assist(t_traindyn *x, void *b, long m, long a, char *s)
{
	if (m == ASSIST_INLET) { //inlet
		sprintf(s, "I am inlet %ld", a);
	}
	else {	// outlet
		sprintf(s, "I am outlet %ld", a);
	}
}


void traindyn_float(t_traindyn *x, double f)
{
}


// registers a function for the signal chain in Max
void traindyn_dsp64(t_traindyn *x, t_object *dsp64, short *count, double samplerate, long maxvectorsize, long flags)
{
    object_method(dsp64, gensym("dsp_add64"), x, traindyn_perform64, 0, NULL);
}


// this is the 64-bit perform method audio vectors
void traindyn_perform64(t_traindyn *x, t_object *dsp64, double **ins, long numins, double **outs, long numouts, long sampleframes, long flags, void *userparam)
{
    if (!x->ready_record) return;
    
    t_double *inL = ins[0];		// we get audio for each inlet of the object from the **ins argument
    t_double *outL = outs[0];	// we get audio for each outlet of the object from the **outs argument
    
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
    
    n = sampleframes;
    
#ifdef FFTW
    if (x->mode==1) //calculate fft from fftw.
    {
        n = sampleframes;
        fftw_plan fft = fftw_plan_dft_r2c_1d(sampleframes, x->time2, x->dft_freq, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
        fftw_execute(fft);
        for(int i=0;i<x->samplesize;i++)
        {
            x->time2[i]=creal(x->dft_freq[i]);
            outL[i]=x->time2[i];
        }
    }
#endif
    
    if (norm > x->threshold || x->rep>0)
    {
        record_training(x, x->samplesize);
    }
}


void record_training(t_traindyn *x, int nframes)
{
    if (x->rep == 0)
    {
        fprintf (x->f_trains, "%d ",x->n_trains);
    }
    
    for(int i=0;i<nframes;i++)
    {
        fprintf (x->f_trains, "%f ",creal(x->time2[i]));
    }
    
    x->rep++;
    
    if (x->rep == x->repeats)
    {
        x->ready_record = false;
        //fprintf (x->f_labels, "\n");
        fprintf (x->f_trains,"\n");
        post("record no. %d",x->n_trains);
        outlet_bang(x->out);
    }
    
}


void new_signal(t_traindyn *x)
{
    if (x->rep<x->repeats || (!x->label_assigned) || !x->flag_loadpath)
    {
        if (x->rep<x->repeats) post("still recording!");
        if (!x->label_assigned) post("assign_label to previous train first!");
        if (!x->flag_loadpath) post("load path first!");
        return; //still recording
    }
    post("record: %d",x->n_trains);
    x->label_assigned=false;
    post("record ready");
    x->ready_record=true;
    x->rep=0;
}


void new_label(t_traindyn *x, t_symbol *s, long ac, t_atom *av)
{
    if (ac>1) //mark active notes
    {
        for (int i=0;i<x->nlabels;i++)
        {
            x->label_vector[i] = 0;
        }
        for (int i=0;i<ac;i++)
        {
            if (atom_getlong(av+i)<x->nlabels)
            {
                post("label set %d",atom_getlong(av+i));
                x->label_vector[atom_getlong(av+i)] =1;
            }
        }
    }
    
    if (x->rep<x->repeats || x->label_assigned)
    {
        post("label changed, waiting for new_signal");
        return; //still recording
    }

    fprintf (x->f_labels, "%d ",x->n_trains);
    for(int i=0;i<x->nlabels;i++)
    {
        //post("%d ",x->label_vector[i]);
        fprintf (x->f_labels, "%d ",x->label_vector[i]);
    }
    fprintf (x->f_labels,"\n");
    post("label no. %d",x->n_trains++);
    x->label_assigned=true;
}


void traindyn_anything(t_traindyn *x, t_symbol *s, long ac, t_atom *av)
{
}


void threshold(t_traindyn *x, float f)
{
    x->threshold = f;
    post("new threshold set");
}






