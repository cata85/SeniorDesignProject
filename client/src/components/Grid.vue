<template>
    <div class="grid-container">
      <imageItem v-for="index in 12"
                 :key="index"
                 :id=index.toString()
                 :imgSrc="getTag(index)"
                 :imgName="getName(index)">
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
      imageCategories: ['Bombus Affinis', 'Bombus Appositus', 'Bombus Auricomus', 'Bombus Bifarius', 'Bombus Bimaculatus', 'Bombus Borealis',
        'Bombus Caliginosus', 'Bombus Centralis', 'Bombus Citrinus', 'Bombus Crotchii', 'Bombus Cryptarum', 'Bombus Fernaldae',
        'Bombus Fervidus', 'Bombus Flavifrons', 'Bombus Fraternus', 'Bombus Frigidus', 'Bombus Griseocollis', 'Bombus Huntii',
        'Bombus Impatiens', 'Bombus Insularis', 'Bombus Melanopygus', 'Bombus Mixtus', 'Bombus Morrisoni', 'Bombus Nevadensis',
        'Bombus Occidentalis', 'Bombus Pensylvanicus', 'Bombus Perplexus', 'Bombus Rufocinctus', 'Bombus Sandersoni',
        'Bombus Sitkensis', 'Bombus Sylvicola', 'Bombus Ternarius', 'Bombus Terricola', 'Bombus Vagans', 'Bombus Vandykei', 'Bombus Vosnesenskii'],
      numOfCategories: 36,
    };
  },
  components: {
    imageItem: Image,
  },
  methods: {
    getTag(index) {
      const tempName = this.imageCategories[this.randInts[index]];
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
    getName(index) {
      return this.imageCategories[this.randInts[index]];
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
    grid-column-gap: 0.5em;
    grid-row-gap: 0.5em;
    justify-content: center;
    align-content: center;
  }
</style>
