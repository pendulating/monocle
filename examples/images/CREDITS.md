# Example images

Carol M. Highsmith took these photos for the Library of Congress. She gave
them to the public domain. We took 640-pixel copies from Wikimedia Commons.

| File | Source | License |
|------|--------|---------|
| `boston_bowfront_houses.jpg` | ["Bowfront houses" in South End, Boston, Massachusetts LCCN2011632380](https://commons.wikimedia.org/wiki/File:%22Bowfront_houses%22_in_South_End,_Boston,_Massachusetts_LCCN2011632380.tif) | Public domain |
| `crawford_downtown.jpg` | ["Downtown" Crawford, Texas, a tiny McLennon County place that swelled during the presidential administration of George W. Bush, who spent much time at his nearby Prairie Chapel Ranch in the early LCCN2015630946](https://commons.wikimedia.org/wiki/File:%22Downtown%22_Crawford,_Texas,_a_tiny_McLennon_County_place_that_swelled_during_the_presidential_administration_of_George_W._Bush,_who_spent_much_time_at_his_nearby_Prairie_Chapel_Ranch_in_the_early_LCCN2015630946.tif) | Public domain |
| `galveston_bishops_palace.jpg` | ["Bishop's Palace," also known as Gresham's Castle, an ornate Victorian-style house, located on Broadway and 14th Street in the East End Historic District of Galveston, Texas LCCN2013650798](https://commons.wikimedia.org/wiki/File:%22Bishop%27s_Palace,%22_also_known_as_Gresham%27s_Castle,_an_ornate_Victorian-style_house,_located_on_Broadway_and_14th_Street_in_the_East_End_Historic_District_of_Galveston,_Texas_LCCN2013650798.tif) | Public domain |
| `gone_fishing_shack.jpg` | ["Gone Fishing" old shack, Seligman, Arizona LCCN2010630120](https://commons.wikimedia.org/wiki/File:%22Gone_Fishing%22_old_shack,_Seligman,_Arizona_LCCN2010630120.tif) | Public domain |
| `hammering_man_seattle.jpg` | ["Hammering Man" art, Seattle, Washington LCCN2011630311](https://commons.wikimedia.org/wiki/File:%22Hammering_Man%22_art,_Seattle,_Washington_LCCN2011630311.tif) | Public domain |
| `main_street_gas.jpg` | ["Main Street Gas" in rural Pocahontas County, West Virginia, has all the accountrements of an old gasoline station. But the closest Main Street of any size is miles away LCCN2015634497](https://commons.wikimedia.org/wiki/File:%22Main_Street_Gas%22_in_rural_Pocahontas_County,_West_Virginia,_has_all_the_accountrements_of_an_old_gasoline_station._But_the_closest_Main_Street_of_any_size_is_miles_away_LCCN2015634497.tif) | Public domain |
| `pharmacy_sign.jpg` | ["Drugs" sign outside a pharmacy on the portion of the National Road that runs through Wheeling, West Virginia LCCN2015632052](https://commons.wikimedia.org/wiki/File:%22Drugs%22_sign_outside_a_pharmacy_on_the_portion_of_the_National_Road_that_runs_through_Wheeling,_West_Virginia_LCCN2015632052.tif) | Public domain |
| `route66_pop_sign.jpg` | ["Ice Cold Pop" sign and American flag advertised on Route 66, Seligman, Arizona LCCN2010630131](https://commons.wikimedia.org/wiki/File:%22Ice_Cold_Pop%22_sign_and_American_flag_advertised_on_Route_66,_Seligman,_Arizona_LCCN2010630131.tif) | Public domain |
| `texas_farm_road.jpg` | ["Forever view" down Texas Farm Road 1111 in Hudspeth County, Texas LCCN2014631666](https://commons.wikimedia.org/wiki/File:%22Forever_view%22_down_Texas_Farm_Road_1111_in_Hudspeth_County,_Texas_LCCN2014631666.tif) | Public domain |
| `zion_great_white_throne.jpg` | ["Great White Throne," Zion National Park, Utah LCCN2011631049](https://commons.wikimedia.org/wiki/File:%22Great_White_Throne,%22_Zion_National_Park,_Utah_LCCN2011631049.tif) | Public domain |

`images.parquet` lists the files with relative paths. Make it again with:

```bash
python scripts/make_image_table.py examples/images examples/images/images.parquet --relative \
    --prompt "Describe this image in one sentence. Then name the type of place it shows."
```
