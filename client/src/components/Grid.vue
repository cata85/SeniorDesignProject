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
      randInts: [],
      numImages: 12,
      imageCategories: ['Bombus affinis', 'Bombus appositus', 'Bombus auricomus', 'Bombus bifarius', 'Bombus bimaculatus', 'Bombus borealis',
        'Bombus caliginosus', 'Bombus centralis', 'Bombus citrinus', 'Bombus crotchii', 'Bombus cryptarum', 'Bombus fernaldae',
        'Bombus fervidus', 'Bombus flavifrons', 'Bombus fraternus', 'Bombus frigidus', 'Bombus griseocollis', 'Bombus huntii',
        'Bombus impatiens', 'Bombus insularis', 'Bombus melanopygus', 'Bombus mixtus', 'Bombus morrisoni', 'Bombus nevadensis',
        'Bombus occidentalis', 'Bombus pensylvanicus', 'Bombus perplexus', 'Bombus rufocinctus', 'Bombus sandersoni',
        'Bombus sitkensis', 'Bombus sylvicola', 'Bombus ternarius', 'Bombus terricola', 'Bombus vagans', 'Bombus vandykei', 'Bombus vosnesenskii'],
      numOfCategories: 36,
    };
  },
  components: {
    imageItem: Image,
  },
  methods: {
    getTag(index) {
      const tempName = this.imageCategories[this.randInts[index]];
      console.log(tempName);
      console.log(this.randInts[index]);
      return './assets/Images/'.concat(tempName).concat('.jpg');
    },
    makeRandom() {
      const arr = [];
      while (arr.length < this.numImages + 1) {
        const r = Math.floor(Math.random() * this.numOfCategories);
        if (arr.indexOf(r) === -1) arr.push(r);
      }
      this.randInts = arr;
    },
  },
  created() {
    this.makeRandom();
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
