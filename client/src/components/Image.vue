<template>
  <div class="grid-item" :id="id" @mouseover="mouseOver" @mouseleave="mouseLeave">
    <img :src="imgSrc" class="_img" :id="getImgId(id)"/>
    <div class="imgText">{{ name }}</div>
  </div>
</template>


<script>
export default {
  props: {
    id: {
      type: String,
      required: true,
    },
    imgSrc: {
      type: String,
      required: true,
    },
  },
  data() {
    return {
      name: '',
      displayName: '',
    };
  },
  methods: {
    getImgId(idVal) {
      return 'img'.concat(idVal);
    },
    mouseOver() {
      const el = this.getElement();
      this.displayName = this.name;
      el.classList.add('imgOver');
    },
    mouseLeave() {
      const el = this.getElement();
      this.displayName = '';
      el.classList.remove('imgOver');
    },
    getElement() {
      const myId = this.getImgId(this.id);
      return document.getElementById(myId);
    },
  },
  mounted() {
    this.$nextTick(function () {
      const el = this.getElement();
      this.name = this.getImgId(this.id);
      setTimeout(() => { el.classList.add('fadeTag'); }, this.id * 200);
    });
  },
};
</script>


<style scoped>
  .grid-item {
    border:0.05em black solid;
    box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19);
    overflow: hidden;
  }

  .imgText {
    text-align: center;
    position: absolute;
    display: inline-block;
    margin: 0 auto;
    z-index: 100;
    font-size: 2em;
    color: white;
  }

  ._img {
    object-fit: cover;
    position: relative;
    width: 100%;
    height: 100%;
    opacity: 0;
  }

  .imgOver {
    -webkit-filter: blur(0.25em); /* Safari 6.0 - 9.0 */
    filter: blur(0.25em);
  }

  .fadeTag {
    -webkit-animation: fadein 2s forwards; /* Safari, Chrome and Opera > 12.1 */
    -moz-animation: fadein 2s forwards; /* Firefox < 16 */
    -ms-animation: fadein 2s forwards; /* Internet Explorer */
      -o-animation: fadein 2s forwards; /* Opera < 12.1 */
        animation: fadein 2s forwards;
  }

  @keyframes fadein {
    from { opacity: 0; }
    to   { opacity: 1; }
  }
</style>
