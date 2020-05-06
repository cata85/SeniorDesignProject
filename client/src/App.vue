<template>
  <div id="app">
    <img id="backgroundImg" src="honeycomb.png"/>
    <router-view/>
  </div>
</template>

<style>
#app {
  margin-top: 60px
}

#backgroundImg {
  height: 100%;
  width: 100%;
  position: absolute;
  top: 0;
  left: 0;
  opacity: 0.1;
}
</style>
