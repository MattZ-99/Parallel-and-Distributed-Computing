<h1><center>Parallel and Distributed Computing Assignment 3<center/></h1>



          姓名：					学号：

#### Experimental environment 

Server GeForce GTX 1080 Ti, Distributed Memory Programming with MPI, based on c.



<h3> Problem 1</h3>

##### Problem discription

1. (round-trip time test) A MPI task 0 is sending beacon packets to another MPI task 1. After receiving each beacon, task 1 will reply with an ACK. Task 0 could calculate the round trip time of the connection. Please simulate this process using MPI.

##### Program execution

Running on shell, compile using make and execution using mpiexec.

Here is my example output:

~~~shell
publichw3_4@ArchLab102:~$ cd zmt
publichw3_4@ArchLab102:~/zmt$ cd problem1
publichw3_4@ArchLab102:~/zmt/problem1$ make
mpicc -o solution1 solution1.c
publichw3_4@ArchLab102:~/zmt/problem1$ mpiexec -n 2 ./solution1
Process 0 of 2 > Send beacon packet. current time: 889904.152418
Process 1 of 2 > ACK. current time: 889904.152466
Process 0 of 2 > Received "ACK." current time: 889904.152491
Beacon send time: 0.000048, ACK receive time: 0.000025, round-trip time: 0.000073
publichw3_4@ArchLab102:~/zmt/problem1$ mpiexec -n 5 ./solution1
Process 4 of 5 > This process is spare.
Process 2 of 5 > This process is spare.
Process 3 of 5 > This process is spare.
Process 0 of 5 > Send beacon packet. current time: 889909.888120
Process 1 of 5 > ACK. current time: 889909.888158
Process 0 of 5 > Received "ACK." current time: 889909.888180
Beacon send time: 0.000037, ACK receive time: 0.000023, round-trip time: 0.000060
publichw3_4@ArchLab102:~/zmt/problem1$ mpiexec -n 1 ./solution1
Process 0 of 1 > Process number error (only 1 process).
publichw3_4@ArchLab102:~/zmt/problem1$ 
~~~

![image-20201028223321998](C:\Users\MengtianZhang\AppData\Roaming\Typora\typora-user-images\image-20201028223321998.png)



##### Core code analysis

The task 0 first send beacon packets to the task 1 and then wait to receive the message from task 1. Finally, task 0 print the time result.

~~~c
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
~~~

And task 1 first receive message from task 0 and then resend ACK to task 0.

~~~c
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
~~~



Attention here. I test the performance of functions $MPI\_Send()$ and $MPI\_Ssend()$.

Time with $MPI\_Send()$:

![image-20201028224937922](C:\Users\MengtianZhang\AppData\Roaming\Typora\typora-user-images\image-20201028224937922.png)

Time with $MPI\_Ssend()$:

![image-20201028224820245](C:\Users\MengtianZhang\AppData\Roaming\Typora\typora-user-images\image-20201028224820245.png)

We can find $MPI\_Send()$ here use less time, as $MPI\_Ssend()$ is guaranteed to block until them matching receive starts. Of course, here both functions can run correctly. But if there exist other outputs in both processes, this function would make sense.




<h3> Problem 2</h3>

##### Problem discription

2. (pass-string problem) Consider some children playing a voice passing game. The child #0 say a sentence to child #1, who will write down the first word and pass remaining words to child #2. Child #2 do the same thing as child #1, write down the first word (which is the second word of the original sentence) and pass the remaining words to next child… The first child with no word received write down his ID.
   Please simulate this process using MPI. (The WRITE DOWN action is substituted by PRINT TO STDOUT)
   
##### Program execution

Running on shell, compile using make and execution using mpiexec.

Here is my example output:

~~~shell
publichw3_4@ArchLab102:~$ cd zmt
publichw3_4@ArchLab102:~/zmt$ cd problem2
publichw3_4@ArchLab102:~/zmt/problem2$ make
mpicc -o solution2 solution2.c
publichw3_4@ArchLab102:~/zmt/problem2$ mpiexec -n 5 ./solution2
Process 1 of 5 > Consider
Process 2 of 5 > some
Process 3 of 5 > children
Process 4 of 5 > playing
Process 0 of 5 > a
Process 1 of 5 > voice
Process 2 of 5 > passing
Process 3 of 5 > game.
Process 4 of 5 > Please
Process 0 of 5 > simulate
Process 1 of 5 > this
Process 2 of 5 > process
Process 3 of 5 > using
Process 4 of 5 > MPI.
Process 0 of 5 > END! my position is 0.
~~~



![image-20201028225916747](C:\Users\MengtianZhang\AppData\Roaming\Typora\typora-user-images\image-20201028225916747.png)



##### Core code analysis

First process need to get the sentence, and pass to the following processes.

~~~c
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
~~~

While loop for processes loop and passing.

~~~c
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
~~~

Sentence split, according to the elements in delim array. 

~~~c
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
~~~

