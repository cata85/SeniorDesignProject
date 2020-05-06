<template>
  <div id="app">
    <div id="backgroundDiv">
      <img id="backgroundImg" src="honeycomb.png"/>
    </div>
    <router-view/>
  </div>
</template>

<style>
#app {
  margin-top: 60px
}

#backgroundDiv {
  position: absolute;
  height: 20%;
  width: 100%;
  top: 0;
  left: 0;
  overflow: hidden;
}

#backgroundImg {
  height: 150%;
  width: 100%;
  -webkit-mask-image: linear-gradient(rgba(0, 0, 0, 1) 0%, rgba(0, 0, 0, 0.5) 0.1%, transparent);
  mask-image: linear-gradient(rgba(0, 0, 0, 1) 0%, rgba(0, 0, 0, 0.5) 0.1%, transparent);
}
</style>
