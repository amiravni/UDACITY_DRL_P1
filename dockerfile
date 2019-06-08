## based on docker with pytorch and jupiter notebook
FROM spellrun/pytorch-jupyter

## get gym enviroment
RUN git clone https://github.com/openai/gym.git
WORKDIR /gym
RUN pip install -e .

## get udacity DRL enviroment
WORKDIR /
RUN git clone https://github.com/udacity/deep-reinforcement-learning.git
WORKDIR deep-reinforcement-learning/python
RUN pip install .

## install Nvidia drivers for unity/opengl support
## NOTE: In order to make it work the drivers must to be the same drivers which are on the host file
#RUN apt-get -y purge nvidia*
RUN add-apt-repository ppa:graphics-drivers
RUN apt-get -y update
RUN apt-get -y install screen
RUN apt-get -y install nvidia-390   #<--- change to the correct number, check with nvidia-smi command


## Optional - install pycharm 
#RUN apt-get install -y openjdk-8-jre-headless
#RUN apt-get install -y ubuntu-make
#RUN umake ide pycharm /root/.local/share/umake/ide/pycharm
#RUN echo "alias pycharm='/root/.local/share/umake/ide/pycharm/bin/pycharm.sh'" >> /root/.bashrc

WORKDIR /deep-reinforcement-learning/p1_navigation/
RUN wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip && unzip Banana_Linux.zip
RUN wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip && unzip Banana_Linux_NoVis.zip
COPY ./TrainedAgents ./TrainedAgents
COPY ./*.ipynb ./
COPY ./*.py ./
COPY ./*.pth ./

WORKDIR /

