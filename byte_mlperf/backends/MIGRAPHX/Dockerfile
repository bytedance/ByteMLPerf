FROM rocm/dev-ubuntu-20.04:5.5

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

WORKDIR /root
ENV HOME /root

# install common dev tools
RUN apt-get -y update \
    && DEBIAN_FRONTEND=noninteractive apt-get -y install --no-install-recommends \
        git \
        wget \
        nano \
        vim \
        # needed for mlperf_loadgen
        libpython3.8-dev \
        # ssh-scankey
        openssh-client \
    # clean up
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# install cmake from github
RUN archive="cmake-3.25.2-linux-x86_64.tar.gz" \
    url="https://github.com/Kitware/CMake/releases/download/v3.25.2/${archive}" \
    && cd /tmp/ \
    && wget --quiet ${url} \
    && tar --strip-components=1 -xzf ${archive} -C /usr/local \
    && rm -rf /tmp/*

# install migraphx and dependencies
RUN apt-get -y update \
    && DEBIAN_FRONTEND=noninteractive apt-get -y install --no-install-recommends \
        rsync \
        # these packages are required to build with migraphx but don't get installed
        # as dependencies. They also create /opt/rocm-* so we can make a symlink
        # before installing migraphx
        miopen-hip-dev \
        rocm-device-libs \
        rocblas-dev \
        python3.8-dev \
        # these packages are required to build AMDMIGraphX from source
        rocm-cmake \
        pkg-config \
	half \
    && echo "/opt/rocm/lib" > /etc/ld.so.conf.d/rocm.conf \
    && echo "/opt/rocm/llvm/lib" > /etc/ld.so.conf.d/rocm-llvm.conf \
    && ldconfig \
    # having symlinks between rocm-* and rocm complicates building the production
    # image so move files over to rocm/ but add a symlink for compatibility
    && dir=$(find /opt/ -maxdepth 1 -type d -name "rocm-*") \
    && rm -rf /opt/rocm \
    && mkdir /opt/rocm \
    && rsync -a $dir/* /opt/rocm/ \
    && rm -rf $dir \
    && ln -s /opt/rocm $dir \
    # clean up
    && apt-get purge -y --auto-remove \
        rsync \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# build AMDMIGraphX development branch from source
RUN pip install https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz && \
    git clone -b develop https://github.com/ROCmSoftwarePlatform/AMDMIGraphX.git && \
    cd AMDMIGraphX && \
    git checkout ea32ca702b1d81ea3bf7bebb8fcd3c897f61df5d && \
    rbuild build -d depend -B build --cxx=/opt/rocm/llvm/bin/clang++ && \
    cd build && make package && dpkg -i *.deb
RUN pip install numpy==1.23.0
