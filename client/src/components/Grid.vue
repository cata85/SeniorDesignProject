<template>
    <div class="grid-container">
      <imageItem v-for="index in 12"
                 :key="index"
                 :id=index.toString()
                 :imgSrc="getTag(index)">
      </imageItem>

    </div>
</template>


<script>
import Image from './Image.vue';

export default {
  data() {
    return {
      images: [],
    };
  },
  components: {
    imageItem: Image,
  },
  methods: {
    getTag(key) {
      return './assets/bumble'.concat(key).concat('.jpg');
    },
  },
};
</script>


<style scoped>
  .grid-container {
    display: grid;
    grid-template-columns: 14em 14em 14em 14em 14em 14em;
    grid-template-rows: 14em 14em;
    /* grid-template-columns: 0.5fr 0.5fr 0.5fr 0.5fr 0.5fr 0.5fr;
    grid-template-rows: 0.06fr 0.06fr; */
    grid-column-gap: 0.5em;
    grid-row-gap: 0.5em;
    justify-content: center;
    align-content: center;
  }
</style>
