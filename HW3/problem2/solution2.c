# include <stdio.h>
#include <stdlib.h>
# include <string.h>
# include <mpi.h>

const int MAX_STRING = 100; // max string size for mpi send and receive

//设置分隔符array，每一个字符均可以作为分隔符
// const char delim[34] = " !\"#$&'()*+,-./:;<=>?@[\\]^_`{|}~%";   
const char delim[2] = " ";

// sentence analysis
// including, string split, sentence output, string copy, dynamic array free.
int sentence_analysis(char * sentence, char ** token, char ** stc, const int my_rank, const int comm_sz) {
    *token = strsep(stc, delim); // string split
    char * head = *token; // head, use to save the origin head pointer, for free
    // while (*token == "") {
    //     *token = strsep(stc, delim);
    // }
    printf("Process %d of %d > %s\n", my_rank, comm_sz, *token); // output result

    if(*stc == NULL) return 1;  // cherk whether sentence is end
    strcpy(sentence, *stc); // copy to the sentence array for MPI send
    free(*token); // free dynamic array
    return 0;
}


// check if flag is valid, and then do different action
void check_flag(int * flag, char * sentence, char ** token, char ** stc, const int my_rank, const int comm_sz) {
    if (*flag == 1) printf("Process %d of %d > END! my position is %d.\n", my_rank, comm_sz, my_rank); 
    if(*flag == 0) *flag = sentence_analysis(sentence, token, stc, my_rank, comm_sz);
    else (*flag)++;
}




int main(void) {

    int comm_sz;
    int my_rank;


    // flag: int. use to mark the sentence situation
    // flag == 0, sentence is not end
    // flag == 1, process is the first after sentence end
    // flag == 2, 3, ... second, third ... 
    // flag > comm_sz, all the processes have gone to end.

    int flag;  // use to record whether there is no word, and the first child no word.
    char sentence[MAX_STRING];  // the pointer used to store string
    char * stc, * token;  // the pointers used to strsep()
     //  define punctuation

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // task 0 initialize the sentence
    if(my_rank == 0) {
        char words[] = "Consider some children playing a voice passing game. Please simulate this process using MPI.";
        // char words[] = "aaa bbb ccc dddaaa bbb ccc dddaaa bbb ccc ddd!";  // initialize sentence
        stc = strdup(words);
        flag = 0;

        // check_flag(&flag, sentence, &token, &stc, my_rank, comm_sz);
        strcpy(sentence, stc);

        if(flag > comm_sz) {
            MPI_Finalize();
        }
        else if (flag > 1) {
            MPI_Ssend(&flag, 1, MPI_INT, (my_rank + 1) % comm_sz, 0, MPI_COMM_WORLD);
            MPI_Finalize();
        }
        else if (flag == 1) {
            MPI_Ssend(&flag, 1, MPI_INT, (my_rank + 1) % comm_sz, 0, MPI_COMM_WORLD);
        }
        else {
            MPI_Ssend(&flag, 1, MPI_INT, (my_rank + 1) % comm_sz, 0, MPI_COMM_WORLD);
            MPI_Ssend(sentence, strlen(sentence) + 1, MPI_CHAR, (my_rank + 1) % comm_sz, 0, MPI_COMM_WORLD); 
        }        
    } 
    
    // loop to pass through all the words
    while(1) {
        // MPI first receive the flag message
        MPI_Recv(&flag, 1, MPI_INT, (my_rank + comm_sz - 1) % comm_sz, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (flag == 0)  { // flag == 0 means, sentence is avaliable
            MPI_Recv(sentence, MAX_STRING, MPI_CHAR, (my_rank + comm_sz - 1) % comm_sz, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            stc = strdup(sentence);
        }
        
        check_flag(&flag, sentence, &token, &stc, my_rank, comm_sz);

        
        if(flag > comm_sz) { // indicates, all the other processes have finalize and break.
            MPI_Finalize();
            break;
        }
        else if (flag > 1) { // need to tell the processes behind to finish, and then break.
            MPI_Ssend(&flag, 1, MPI_INT, (my_rank + 1) % comm_sz, 0, MPI_COMM_WORLD);
            MPI_Finalize();
            break;
        }
        else if (flag == 1) {// the first process without word.
            MPI_Ssend(&flag, 1, MPI_INT, (my_rank + 1) % comm_sz, 0, MPI_COMM_WORLD);
        }
        else { // flag == 0. valid sentence, pass flag and sentence
            MPI_Ssend(&flag, 1, MPI_INT, (my_rank + 1) % comm_sz, 0, MPI_COMM_WORLD);
            MPI_Ssend(sentence, strlen(sentence) + 1, MPI_CHAR, (my_rank + 1) % comm_sz, 0, MPI_COMM_WORLD);   
        }  
        
    }
    
    return 0;

}

