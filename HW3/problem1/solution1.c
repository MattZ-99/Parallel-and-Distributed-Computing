# include <stdio.h>
# include <string.h>
# include <mpi.h>

const int MAX_STRING = 100;   // max string size for mpi send and receive

int main(void) {
    char beacon[MAX_STRING];  // array for beacon packets
    char ack[MAX_STRING];   // array for ack message
    int comm_sz;
    int my_rank;
    double start, mid, finish; // for timer


    // MPI initialize
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // deal with invalid processes number.
    if(comm_sz == 1) {
        printf("Process %d of %d > Process number error (only 1 process).\n", my_rank, comm_sz);
        MPI_Finalize();
        return 0;
    }
    if(my_rank > 1) printf("Process %d of %d > This process is spare.\n", my_rank, comm_sz);
    
    MPI_Barrier(MPI_COMM_WORLD);


    // round-trip between task 0 and 1.

    if(my_rank == 0) {
        
        sprintf(beacon, "Send beacon packet.");
        start = MPI_Wtime();
        printf("Process %d of %d > %s current time: %f\n", my_rank, comm_sz, beacon, start);
        // send messages. and I compare MPI_Ssend() and MPI_Send() here.
        // two messages. first string, and second double for time.
        MPI_Ssend(beacon, strlen(beacon) + 1, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
        MPI_Ssend(&start, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
        
        //receive messages.
        MPI_Recv(ack, MAX_STRING, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&mid, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // SEndrecv() function test.
        // MPI_Sendrecv(beacon, strlen(beacon) + 1, MPI_CHAR, 1, 0, 
        //             ack, MAX_STRING, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);


        finish = MPI_Wtime();
        // output final results
        printf("Process %d of %d > Received \"%s\" current time: %f\n", my_rank, comm_sz, ack, finish);
        printf("Beacon send time: %f, ACK receive time: %f, round-trip time: %f\n", mid-start, finish-mid, finish-start);
    }

    if( my_rank == 1) {
        // receive messages from task 0
        MPI_Recv(beacon, MAX_STRING, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&start, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // record mid time
        mid = MPI_Wtime();
        sprintf(ack, "ACK.");
        printf("Process %d of %d > %s current time: %f\n", my_rank, comm_sz, ack, mid);

        //Send ACK messages back
        MPI_Ssend(ack, strlen(ack) + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        MPI_Ssend(&mid, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;

}