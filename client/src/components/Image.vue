<template>
  <div class="grid-item" :id="id" @mouseover="mouseOver" @mouseleave="mouseLeave">
    <img :src="imgSrc" :name="imgName" class="_img" :id="getImgId(id)"/>
    <div class="imgText">{{ displayName }}</div>
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
    getImgId(idVal) {
      return 'img'.concat(idVal);
    },
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
      const myId = this.getImgId(this.id);
      return document.getElementById(myId);
    },
  },
  mounted() {
    this.$nextTick(function nextTick() {
      const el = this.getElement();
      setTimeout(() => { el.classList.add('fadeTag'); }, this.id * 100);
    });
  },
};
</script>


<style scoped>
  .grid-item {
    box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19);
    overflow: hidden;
    display: grid;
    width: 100%;
    position: relative;
  }

  .imgText {
    z-index: 100;
    font-size: 2em;
    color: white;
    margin: 0 auto;
    grid-row: 1;
    grid-column: 1;
    align-self: center;
    text-align: center;
  }

  ._img {
    grid-row: 1;
    grid-column: 1;
    object-fit: cover;
    /* position: relative; */
    display: block;
    width: 100%;
    height: 100%;
    opacity: 0;
  }

  .imgOver {
    -webkit-filter: blur(0.25em); /* Safari 6.0 - 9.0 */
    filter: blur(0.25em) brightness(50%);
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
