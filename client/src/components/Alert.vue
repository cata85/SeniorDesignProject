<template>
  <div>
    <b-alert variant="success" show center-block class='alert-grid'>
      <img :src="imgSrc" :name="imgName" :id="imgId" class="_img"
        @mouseover="mouseOver" @mouseleave="mouseLeave"/>
      <div class="imgText">{{ displayName }}</div>
    </b-alert>
    <br>
  </div>
</template>

<script>
export default {
  props: {
    imgId: {
      type: String,
      required: true,
    },
    imgSrc: {
      type: String,
      required: true,
    },
    imgName: {
      type: String,
      required: true,
    },
  },
  data() {
    return {
      displayName: '',
    };
  },
  methods: {
    mouseOver() {
      const el = this.getElement();
      this.displayName = this.imgName;
      el.classList.add('imgOver');
    },
    mouseLeave() {
      const el = this.getElement();
      this.displayName = '';
      el.classList.remove('imgOver');
    },
    getElement() {
      return document.getElementById(this.imgId);
    },
  },
};
</script>


<style scoped>
  .alert-grid {
    box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19);
    overflow: hidden;
    width: 30%;
    text-align: center;
    position: relative;
    margin: auto;
  }

  .imgText {
    z-index: 100;
    font-size: 2em;
    color: white;
    position: absolute;
    top: 50%;
    left: 50%;
    -webkit-transform: translate(-50%, -50%);
    -ms-transform: translate(-50%, -50%);
    transform: translate(-50%, -50%);
    text-align: center;
  }

  ._img {
    object-fit: cover;
    position: relative;
    display: block;
    width: 100%;
    height: 100%;
  }

  .imgOver {
    -webkit-filter: blur(0.25em); /* Safari 6.0 - 9.0 */
    filter: blur(0.25em) brightness(50%);
  }
</style>
