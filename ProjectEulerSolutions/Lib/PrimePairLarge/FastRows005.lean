import ProjectEulerSolutions.Lib.PrimePairLarge.Data
import ProjectEulerSolutions.Lib.SortedNatTree
namespace ProjectEulerSolutions.PrimePairLarge.FastRows
def row2141 : SortedNatTree.Tree := (.node 9623 (.node 5153 (.node 3191 (.node 2621 (.node 2309 (.node 2267 .empty .empty) .empty) (.node 3167 (.node 2663 .empty .empty) .empty)) (.node 4049 (.node 3947 (.node 3881 .empty .empty) .empty) (.node 4649 (.node 4091 .empty .empty) .empty))) (.node 7691 (.node 6737 (.node 6053 (.node 5519 .empty .empty) .empty) (.node 7547 (.node 7349 .empty .empty) .empty)) (.node 8951 (.node 8693 (.node 7727 .empty .empty) .empty) (.node 9551 .empty .empty)))) (.node 17627 (.node 13901 (.node 11813 (.node 11351 (.node 10139 .empty .empty) .empty) (.node 12281 (.node 11903 .empty .empty) .empty)) (.node 15641 (.node 14969 (.node 14951 .empty .empty) .empty) (.node 17207 (.node 16127 .empty .empty) .empty))) (.node 20903 (.node 19421 (.node 19013 (.node 18413 .empty .empty) .empty) (.node 20789 (.node 20261 .empty .empty) .empty)) (.node 22751 (.node 21893 (.node 21269 .empty .empty) .empty) (.node 23549 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2141_ordered : SortedNatTree.ordered row2141 = true := by decide
set_option maxRecDepth 4096 in
theorem row2141_values : SortedNatTree.values row2141 = Data.row2141 := rfl
theorem row2141_checked : SortedNatTree.contains row2141 = Data.row2141.contains := by
  rw [SortedNatTree.contains_eq_list row2141 row2141_ordered, row2141_values]
def row2143 : SortedNatTree.Tree := (.node 13999 (.node 8017 (.node 6229 (.node 4657 (.node 2347 (.node 2239 .empty .empty) .empty) (.node 4801 .empty .empty)) (.node 6691 (.node 6301 .empty .empty) (.node 6871 .empty .empty))) (.node 11941 (.node 9109 (.node 8863 (.node 8089 .empty .empty) .empty) (.node 9283 .empty .empty)) (.node 13381 (.node 12613 .empty .empty) (.node 13399 .empty .empty)))) (.node 20731 (.node 18523 (.node 17137 (.node 15427 (.node 14629 .empty .empty) .empty) (.node 17989 .empty .empty)) (.node 19891 (.node 19603 .empty .empty) (.node 20521 .empty .empty))) (.node 22063 (.node 21937 (.node 20959 (.node 20749 .empty .empty) .empty) (.node 22039 .empty .empty)) (.node 22861 (.node 22777 .empty .empty) (.node 23251 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2143_ordered : SortedNatTree.ordered row2143 = true := by decide
set_option maxRecDepth 4096 in
theorem row2143_values : SortedNatTree.values row2143 = Data.row2143 := rfl
theorem row2143_checked : SortedNatTree.contains row2143 = Data.row2143.contains := by
  rw [SortedNatTree.contains_eq_list row2143 row2143_ordered, row2143_values]
def row2153 : SortedNatTree.Tree := (.node 13781 (.node 6869 (.node 5237 (.node 2969 (.node 2663 (.node 2417 .empty .empty) .empty) (.node 4967 (.node 4397 .empty .empty) .empty)) (.node 6143 (.node 5351 (.node 5273 .empty .empty) .empty) (.node 6863 .empty .empty))) (.node 9839 (.node 9137 (.node 7151 (.node 7121 .empty .empty) .empty) (.node 9209 .empty .empty)) (.node 11897 (.node 10853 (.node 10301 .empty .empty) .empty) (.node 13001 .empty .empty)))) (.node 19301 (.node 17321 (.node 14621 (.node 14207 (.node 13997 .empty .empty) .empty) (.node 16103 .empty .empty)) (.node 18911 (.node 18041 (.node 17579 .empty .empty) .empty) (.node 19157 .empty .empty))) (.node 20477 (.node 20123 (.node 19541 (.node 19463 .empty .empty) .empty) (.node 20219 .empty .empty)) (.node 22871 (.node 21569 (.node 20981 .empty .empty) .empty) (.node 23189 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2153_ordered : SortedNatTree.ordered row2153 = true := by decide
set_option maxRecDepth 4096 in
theorem row2153_values : SortedNatTree.values row2153 = Data.row2153 := rfl
theorem row2153_checked : SortedNatTree.contains row2153 = Data.row2153.contains := by
  rw [SortedNatTree.contains_eq_list row2153 row2153_ordered, row2153_values]
def row2161 : SortedNatTree.Tree := (.node 10177 (.node 6949 (.node 6079 (.node 4597 (.node 4357 (.node 2887 .empty .empty) .empty) (.node 5479 .empty .empty)) (.node 6373 (.node 6337 (.node 6211 .empty .empty) .empty) (.node 6841 .empty .empty))) (.node 7681 (.node 7411 (.node 7237 (.node 7027 .empty .empty) .empty) (.node 7621 .empty .empty)) (.node 9721 (.node 9127 (.node 7759 .empty .empty) .empty) (.node 9883 .empty .empty)))) (.node 18457 (.node 13147 (.node 12511 (.node 11437 (.node 10729 .empty .empty) .empty) (.node 12781 .empty .empty)) (.node 16273 (.node 14929 (.node 14629 .empty .empty) .empty) (.node 18289 .empty .empty))) (.node 21937 (.node 20551 (.node 19819 (.node 19417 .empty .empty) .empty) (.node 21559 .empty .empty)) (.node 22441 (.node 22279 .empty .empty) (.node 22639 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2161_ordered : SortedNatTree.ordered row2161 = true := by decide
set_option maxRecDepth 4096 in
theorem row2161_values : SortedNatTree.values row2161 = Data.row2161 := rfl
theorem row2161_checked : SortedNatTree.contains row2161 = Data.row2161.contains := by
  rw [SortedNatTree.contains_eq_list row2161 row2161_ordered, row2161_values]
def row2179 : SortedNatTree.Tree := (.node 11503 (.node 5857 (.node 3631 (.node 3169 (.node 2917 (.node 2503 .empty .empty) .empty) (.node 3469 (.node 3253 .empty .empty) .empty)) (.node 5227 (.node 5119 (.node 4261 .empty .empty) .empty) (.node 5647 .empty .empty))) (.node 9109 (.node 8419 (.node 6529 (.node 6343 .empty .empty) .empty) (.node 8467 .empty .empty)) (.node 9871 (.node 9781 (.node 9241 .empty .empty) .empty) (.node 10453 .empty .empty)))) (.node 18289 (.node 16843 (.node 14503 (.node 13789 (.node 11731 .empty .empty) .empty) (.node 15559 (.node 14929 .empty .empty) .empty)) (.node 17053 (.node 16921 (.node 16903 .empty .empty) .empty) (.node 17683 .empty .empty))) (.node 21061 (.node 19927 (.node 19423 (.node 18517 .empty .empty) .empty) (.node 20719 .empty .empty)) (.node 22051 (.node 21817 (.node 21673 .empty .empty) .empty) (.node 23053 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2179_ordered : SortedNatTree.ordered row2179 = true := by decide
set_option maxRecDepth 4096 in
theorem row2179_values : SortedNatTree.values row2179 = Data.row2179 := rfl
theorem row2179_checked : SortedNatTree.contains row2179 = Data.row2179.contains := by
  rw [SortedNatTree.contains_eq_list row2179 row2179_ordered, row2179_values]
def row2203 : SortedNatTree.Tree := (.node 11863 (.node 6529 (.node 5623 (.node 3061 (.node 2887 (.node 2371 .empty .empty) .empty) (.node 5557 (.node 5521 .empty .empty) .empty)) (.node 6037 (.node 5857 (.node 5647 .empty .empty) .empty) (.node 6271 .empty .empty))) (.node 9133 (.node 7927 (.node 7789 (.node 6961 .empty .empty) .empty) (.node 8389 (.node 8161 .empty .empty) .empty)) (.node 10243 (.node 10141 (.node 9697 .empty .empty) .empty) (.node 11353 .empty .empty)))) (.node 18517 (.node 14821 (.node 12487 (.node 12409 (.node 12241 .empty .empty) .empty) (.node 14449 (.node 14251 .empty .empty) .empty)) (.node 17599 (.node 17389 (.node 16273 .empty .empty) .empty) (.node 18493 .empty .empty))) (.node 21523 (.node 20407 (.node 19483 (.node 19207 .empty .empty) .empty) (.node 21499 (.node 21187 .empty .empty) .empty)) (.node 22573 (.node 21589 (.node 21559 .empty .empty) .empty) (.node 23071 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2203_ordered : SortedNatTree.ordered row2203 = true := by decide
set_option maxRecDepth 4096 in
theorem row2203_values : SortedNatTree.values row2203 = Data.row2203 := rfl
theorem row2203_checked : SortedNatTree.contains row2203 = Data.row2203.contains := by
  rw [SortedNatTree.contains_eq_list row2203 row2203_ordered, row2203_values]
def row2207 : SortedNatTree.Tree := (.node 12251 (.node 6563 (.node 3929 (.node 2777 (.node 2753 (.node 2339 .empty .empty) .empty) (.node 3347 (.node 2819 .empty .empty) .empty)) (.node 5171 (.node 4337 (.node 4271 .empty .empty) .empty) (.node 6053 .empty .empty))) (.node 9221 (.node 8681 (.node 8363 (.node 7193 .empty .empty) .empty) (.node 9137 .empty .empty)) (.node 11801 (.node 10247 (.node 9437 .empty .empty) .empty) (.node 12161 .empty .empty)))) (.node 16547 (.node 15269 (.node 14783 (.node 12983 (.node 12917 .empty .empty) .empty) (.node 15017 (.node 14939 .empty .empty) .empty)) (.node 16127 (.node 16091 (.node 15581 .empty .empty) .empty) (.node 16421 .empty .empty))) (.node 20543 (.node 19157 (.node 19139 (.node 18521 .empty .empty) .empty) (.node 20369 .empty .empty)) (.node 22637 (.node 22013 (.node 21401 .empty .empty) .empty) (.node 23339 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2207_ordered : SortedNatTree.ordered row2207 = true := by decide
set_option maxRecDepth 4096 in
theorem row2207_values : SortedNatTree.values row2207 = Data.row2207 := rfl
theorem row2207_checked : SortedNatTree.contains row2207 = Data.row2207.contains := by
  rw [SortedNatTree.contains_eq_list row2207 row2207_ordered, row2207_values]
def row2213 : SortedNatTree.Tree := (.node 12671 (.node 7547 (.node 4691 (.node 4253 (.node 3581 (.node 2711 .empty .empty) .empty) (.node 4451 .empty .empty)) (.node 6029 (.node 5417 .empty .empty) (.node 7457 .empty .empty))) (.node 9419 (.node 8147 (.node 8093 .empty .empty) (.node 8243 .empty .empty)) (.node 11813 (.node 10091 .empty .empty) (.node 12251 .empty .empty)))) (.node 20219 (.node 14669 (.node 14447 (.node 13241 (.node 12809 .empty .empty) .empty) (.node 14519 .empty .empty)) (.node 18059 (.node 15767 .empty .empty) (.node 19181 .empty .empty))) (.node 21647 (.node 20849 (.node 20681 .empty .empty) (.node 21323 .empty .empty)) (.node 22901 (.node 22397 .empty .empty) (.node 23741 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2213_ordered : SortedNatTree.ordered row2213 = true := by decide
set_option maxRecDepth 4096 in
theorem row2213_values : SortedNatTree.values row2213 = Data.row2213 := rfl
theorem row2213_checked : SortedNatTree.contains row2213 = Data.row2213.contains := by
  rw [SortedNatTree.contains_eq_list row2213 row2213_ordered, row2213_values]
def row2221 : SortedNatTree.Tree := (.node 9283 (.node 5581 (.node 4513 (.node 3847 (.node 3049 (.node 2389 .empty .empty) .empty) (.node 4201 (.node 4111 .empty .empty) .empty)) (.node 4909 (.node 4663 (.node 4519 .empty .empty) .empty) (.node 5527 .empty .empty))) (.node 6673 (.node 6043 (.node 5839 (.node 5737 .empty .empty) .empty) (.node 6451 .empty .empty)) (.node 7873 (.node 7681 (.node 7639 .empty .empty) .empty) (.node 8059 .empty .empty)))) (.node 15601 (.node 13033 (.node 10459 (.node 9973 (.node 9907 .empty .empty) .empty) (.node 12829 .empty .empty)) (.node 14083 (.node 13999 (.node 13711 .empty .empty) .empty) (.node 14713 .empty .empty))) (.node 20161 (.node 20029 (.node 19699 (.node 15919 .empty .empty) .empty) (.node 20047 .empty .empty)) (.node 22189 (.node 20641 (.node 20341 .empty .empty) .empty) (.node 22279 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2221_ordered : SortedNatTree.ordered row2221 = true := by decide
set_option maxRecDepth 4096 in
theorem row2221_values : SortedNatTree.values row2221 = Data.row2221 := rfl
theorem row2221_checked : SortedNatTree.contains row2221 = Data.row2221.contains := by
  rw [SortedNatTree.contains_eq_list row2221 row2221_ordered, row2221_values]
def row2237 : SortedNatTree.Tree := (.node 10211 (.node 4637 (.node 4019 (.node 3251 (.node 2411 (.node 2381 .empty .empty) (.node 2549 .empty .empty)) (.node 3929 (.node 3257 .empty .empty) .empty)) (.node 4349 (.node 4139 (.node 4049 .empty .empty) .empty) (.node 4481 (.node 4421 .empty .empty) .empty))) (.node 6779 (.node 6659 (.node 6581 (.node 5693 .empty .empty) .empty) (.node 6701 (.node 6689 .empty .empty) .empty)) (.node 8423 (.node 6917 (.node 6791 .empty .empty) .empty) (.node 9431 (.node 9419 .empty .empty) .empty)))) (.node 17789 (.node 13613 (.node 11987 (.node 11981 (.node 11261 .empty .empty) .empty) (.node 13241 (.node 12149 .empty .empty) .empty)) (.node 16091 (.node 15923 (.node 14831 .empty .empty) .empty) (.node 16937 (.node 16901 .empty .empty) .empty))) (.node 21317 (.node 19553 (.node 18959 (.node 18251 .empty .empty) .empty) (.node 21089 (.node 19997 .empty .empty) .empty)) (.node 22571 (.node 22229 (.node 21821 .empty .empty) .empty) (.node 23399 (.node 23333 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row2237_ordered : SortedNatTree.ordered row2237 = true := by decide
set_option maxRecDepth 4096 in
theorem row2237_values : SortedNatTree.values row2237 = Data.row2237 := rfl
theorem row2237_checked : SortedNatTree.contains row2237 = Data.row2237.contains := by
  rw [SortedNatTree.contains_eq_list row2237 row2237_ordered, row2237_values]
def row2239 : SortedNatTree.Tree := (.node 9439 (.node 6481 (.node 4987 (.node 4129 (.node 3853 (.node 2857 .empty .empty) (.node 3919 .empty .empty)) (.node 4441 (.node 4261 .empty .empty) .empty)) (.node 5557 (.node 5209 (.node 5179 .empty .empty) .empty) (.node 6007 (.node 5683 .empty .empty) .empty))) (.node 7951 (.node 7351 (.node 7027 (.node 6823 .empty .empty) (.node 7237 .empty .empty)) (.node 7411 (.node 7393 .empty .empty) .empty)) (.node 8419 (.node 8329 (.node 8221 .empty .empty) .empty) (.node 9283 (.node 8923 .empty .empty) .empty)))) (.node 17959 (.node 12433 (.node 11719 (.node 11329 (.node 10069 .empty .empty) (.node 11689 .empty .empty)) (.node 12211 (.node 11953 .empty .empty) .empty)) (.node 17191 (.node 15193 (.node 15187 .empty .empty) .empty) (.node 17713 (.node 17389 .empty .empty) .empty))) (.node 21391 (.node 19249 (.node 18523 (.node 18379 .empty .empty) .empty) (.node 20563 (.node 19477 .empty .empty) .empty)) (.node 21937 (.node 21589 (.node 21487 .empty .empty) .empty) (.node 23071 (.node 23041 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row2239_ordered : SortedNatTree.ordered row2239 = true := by decide
set_option maxRecDepth 4096 in
theorem row2239_values : SortedNatTree.values row2239 = Data.row2239 := rfl
theorem row2239_checked : SortedNatTree.contains row2239 = Data.row2239.contains := by
  rw [SortedNatTree.contains_eq_list row2239 row2239_ordered, row2239_values]
def row2243 : SortedNatTree.Tree := (.node 12527 (.node 6977 (.node 4457 (.node 2789 (.node 2411 (.node 2273 .empty .empty) .empty) (.node 4133 (.node 3539 .empty .empty) .empty)) (.node 6599 (.node 5927 (.node 5477 .empty .empty) .empty) (.node 6899 .empty .empty))) (.node 9377 (.node 8081 (.node 7487 (.node 7193 .empty .empty) .empty) (.node 8573 .empty .empty)) (.node 10133 (.node 10061 (.node 9497 .empty .empty) .empty) (.node 10433 .empty .empty)))) (.node 18251 (.node 16487 (.node 16349 (.node 13127 (.node 12743 .empty .empty) .empty) (.node 16427 .empty .empty)) (.node 17189 (.node 17033 (.node 16931 .empty .empty) .empty) (.node 18191 .empty .empty))) (.node 21191 (.node 19793 (.node 19457 (.node 18797 .empty .empty) .empty) (.node 21107 .empty .empty)) (.node 22643 (.node 21893 (.node 21683 .empty .empty) .empty) (.node 23459 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2243_ordered : SortedNatTree.ordered row2243 = true := by decide
set_option maxRecDepth 4096 in
theorem row2243_values : SortedNatTree.values row2243 = Data.row2243 := rfl
theorem row2243_checked : SortedNatTree.contains row2243 = Data.row2243.contains := by
  rw [SortedNatTree.contains_eq_list row2243 row2243_ordered, row2243_values]
def row2251 : SortedNatTree.Tree := (.node 9787 (.node 6451 (.node 4111 (.node 3307 (.node 3217 (.node 2593 .empty .empty) .empty) (.node 4021 .empty .empty)) (.node 5503 (.node 4201 .empty .empty) (.node 5881 .empty .empty))) (.node 8233 (.node 7927 (.node 7669 (.node 6841 .empty .empty) .empty) (.node 8191 .empty .empty)) (.node 9349 (.node 9103 .empty .empty) (.node 9613 .empty .empty)))) (.node 14683 (.node 12541 (.node 12157 (.node 11839 (.node 10939 .empty .empty) .empty) (.node 12457 .empty .empty)) (.node 13219 (.node 13003 .empty .empty) (.node 13417 .empty .empty))) (.node 21313 (.node 19717 (.node 16651 (.node 15271 .empty .empty) .empty) (.node 20641 .empty .empty)) (.node 23593 (.node 22039 .empty .empty) (.node 23719 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2251_ordered : SortedNatTree.ordered row2251 = true := by decide
set_option maxRecDepth 4096 in
theorem row2251_values : SortedNatTree.values row2251 = Data.row2251 := rfl
theorem row2251_checked : SortedNatTree.contains row2251 = Data.row2251.contains := by
  rw [SortedNatTree.contains_eq_list row2251 row2251_ordered, row2251_values]
def row2267 : SortedNatTree.Tree := (.node 9227 (.node 6053 (.node 4721 (.node 3323 (.node 2939 (.node 2729 .empty .empty) .empty) (.node 4637 (.node 3527 .empty .empty) .empty)) (.node 5867 (.node 5843 (.node 5669 .empty .empty) .empty) (.node 5897 (.node 5879 .empty .empty) .empty))) (.node 8147 (.node 6833 (.node 6791 (.node 6389 .empty .empty) .empty) (.node 7673 (.node 6917 .empty .empty) .empty)) (.node 8819 (.node 8741 (.node 8219 .empty .empty) .empty) (.node 9203 (.node 8861 .empty .empty) .empty)))) (.node 16481 (.node 12041 (.node 9551 (.node 9413 (.node 9257 .empty .empty) .empty) (.node 10973 (.node 10331 .empty .empty) .empty)) (.node 13259 (.node 12659 (.node 12611 .empty .empty) .empty) (.node 15551 (.node 15497 .empty .empty) .empty))) (.node 19937 (.node 18269 (.node 18257 (.node 16883 .empty .empty) .empty) (.node 19841 (.node 18731 .empty .empty) .empty)) (.node 22751 (.node 21839 (.node 21683 .empty .empty) .empty) (.node 23687 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2267_ordered : SortedNatTree.ordered row2267 = true := by decide
set_option maxRecDepth 4096 in
theorem row2267_values : SortedNatTree.values row2267 = Data.row2267 := rfl
theorem row2267_checked : SortedNatTree.contains row2267 = Data.row2267.contains := by
  rw [SortedNatTree.contains_eq_list row2267 row2267_ordered, row2267_values]
def row2269 : SortedNatTree.Tree := (.node 9631 (.node 5923 (.node 4621 (.node 3613 (.node 3067 (.node 2281 .empty .empty) (.node 3373 .empty .empty)) (.node 4057 (.node 3739 .empty .empty) .empty)) (.node 5107 (.node 4909 (.node 4663 .empty .empty) .empty) (.node 5821 (.node 5347 .empty .empty) .empty))) (.node 8563 (.node 7561 (.node 7237 (.node 7027 .empty .empty) (.node 7489 .empty .empty)) (.node 8443 (.node 7879 .empty .empty) .empty)) (.node 9151 (.node 9109 (.node 8779 .empty .empty) .empty) (.node 9337 (.node 9181 .empty .empty) .empty)))) (.node 17077 (.node 11941 (.node 10429 (.node 9949 (.node 9697 .empty .empty) (.node 10093 .empty .empty)) (.node 11131 (.node 11071 .empty .empty) .empty)) (.node 15373 (.node 13633 (.node 13339 .empty .empty) .empty) (.node 16111 (.node 15427 .empty .empty) .empty))) (.node 21391 (.node 20521 (.node 17881 (.node 17191 .empty .empty) (.node 18379 .empty .empty)) (.node 21379 (.node 21001 .empty .empty) .empty)) (.node 23227 (.node 22483 (.node 21727 .empty .empty) .empty) (.node 23761 (.node 23269 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row2269_ordered : SortedNatTree.ordered row2269 = true := by decide
set_option maxRecDepth 4096 in
theorem row2269_values : SortedNatTree.values row2269 = Data.row2269 := rfl
theorem row2269_checked : SortedNatTree.contains row2269 = Data.row2269.contains := by
  rw [SortedNatTree.contains_eq_list row2269 row2269_ordered, row2269_values]
def row2273 : SortedNatTree.Tree := (.node 11321 (.node 7283 (.node 3761 (.node 2999 (.node 2789 (.node 2609 .empty .empty) (.node 2903 .empty .empty)) (.node 3461 (.node 3041 .empty .empty) .empty)) (.node 5711 (.node 5399 (.node 4973 .empty .empty) .empty) (.node 5897 (.node 5861 .empty .empty) .empty))) (.node 9323 (.node 8627 (.node 8297 (.node 8147 .empty .empty) .empty) (.node 8861 (.node 8669 .empty .empty) .empty)) (.node 10739 (.node 10037 (.node 9791 .empty .empty) .empty) (.node 11087 (.node 11027 .empty .empty) .empty)))) (.node 18353 (.node 14369 (.node 12689 (.node 12227 (.node 11597 .empty .empty) .empty) (.node 14057 (.node 12809 .empty .empty) .empty)) (.node 16253 (.node 15017 (.node 14387 .empty .empty) .empty) (.node 16787 (.node 16553 .empty .empty) .empty))) (.node 20543 (.node 19181 (.node 19013 (.node 18797 .empty .empty) .empty) (.node 20147 (.node 19997 .empty .empty) .empty)) (.node 21911 (.node 21023 (.node 20963 .empty .empty) .empty) (.node 23333 (.node 22259 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row2273_ordered : SortedNatTree.ordered row2273 = true := by decide
set_option maxRecDepth 4096 in
theorem row2273_values : SortedNatTree.values row2273 = Data.row2273 := rfl
theorem row2273_checked : SortedNatTree.contains row2273 = Data.row2273.contains := by
  rw [SortedNatTree.contains_eq_list row2273 row2273_ordered, row2273_values]
def row2281 : SortedNatTree.Tree := (.node 10459 (.node 6481 (.node 4273 (.node 2887 (.node 2731 (.node 2551 .empty .empty) .empty) (.node 4201 .empty .empty)) (.node 6361 (.node 5857 (.node 4591 .empty .empty) .empty) (.node 6373 .empty .empty))) (.node 9091 (.node 8179 (.node 7351 (.node 6793 .empty .empty) .empty) (.node 8863 .empty .empty)) (.node 9649 (.node 9127 .empty .empty) (.node 9871 .empty .empty)))) (.node 18973 (.node 14629 (.node 12373 (.node 11197 (.node 10627 .empty .empty) .empty) (.node 12853 .empty .empty)) (.node 18169 (.node 17581 (.node 17443 .empty .empty) .empty) (.node 18691 .empty .empty))) (.node 21031 (.node 20359 (.node 20173 (.node 19699 .empty .empty) .empty) (.node 20983 .empty .empty)) (.node 21871 (.node 21319 .empty .empty) (.node 23209 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2281_ordered : SortedNatTree.ordered row2281 = true := by decide
set_option maxRecDepth 4096 in
theorem row2281_values : SortedNatTree.values row2281 = Data.row2281 := rfl
theorem row2281_checked : SortedNatTree.contains row2281 = Data.row2281.contains := by
  rw [SortedNatTree.contains_eq_list row2281 row2281_ordered, row2281_values]
def row2287 : SortedNatTree.Tree := (.node 9613 (.node 5077 (.node 3943 (.node 3187 (.node 2683 (.node 2341 .empty .empty) .empty) (.node 3607 (.node 3457 .empty .empty) .empty)) (.node 4861 (.node 4483 (.node 4339 .empty .empty) .empty) (.node 5059 (.node 4933 .empty .empty) .empty))) (.node 7681 (.node 6277 (.node 5857 (.node 5431 .empty .empty) .empty) (.node 7537 (.node 6793 .empty .empty) .empty)) (.node 8377 (.node 8221 (.node 8101 .empty .empty) .empty) (.node 9049 (.node 9007 .empty .empty) .empty)))) (.node 16843 (.node 12049 (.node 11173 (.node 10663 (.node 10453 .empty .empty) .empty) (.node 11503 (.node 11437 .empty .empty) .empty)) (.node 14449 (.node 13999 (.node 12553 .empty .empty) .empty) (.node 16231 (.node 14713 .empty .empty) .empty))) (.node 21481 (.node 20641 (.node 19423 (.node 18013 .empty .empty) .empty) (.node 21313 (.node 20707 .empty .empty) .empty)) (.node 22147 (.node 22129 (.node 21841 .empty .empty) .empty) (.node 23371 (.node 22861 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row2287_ordered : SortedNatTree.ordered row2287 = true := by decide
set_option maxRecDepth 4096 in
theorem row2287_values : SortedNatTree.values row2287 = Data.row2287 := rfl
theorem row2287_checked : SortedNatTree.contains row2287 = Data.row2287.contains := by
  rw [SortedNatTree.contains_eq_list row2287 row2287_ordered, row2287_values]
def row2293 : SortedNatTree.Tree := (.node 9199 (.node 5749 (.node 3853 (.node 2683 (.node 2671 .empty .empty) (.node 3727 .empty .empty)) (.node 5281 (.node 5197 .empty .empty) (.node 5569 .empty .empty))) (.node 7333 (.node 6829 (.node 5827 .empty .empty) (.node 7039 .empty .empty)) (.node 9187 (.node 8011 .empty .empty) .empty))) (.node 17827 (.node 14563 (.node 12211 (.node 9277 .empty .empty) (.node 13093 .empty .empty)) (.node 16447 (.node 14851 .empty .empty) (.node 17467 .empty .empty))) (.node 21019 (.node 19309 (.node 18517 .empty .empty) (.node 20353 .empty .empty)) (.node 23671 (.node 21943 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row2293_ordered : SortedNatTree.ordered row2293 = true := by decide
set_option maxRecDepth 4096 in
theorem row2293_values : SortedNatTree.values row2293 = Data.row2293 := rfl
theorem row2293_checked : SortedNatTree.contains row2293 = Data.row2293.contains := by
  rw [SortedNatTree.contains_eq_list row2293 row2293_ordered, row2293_values]
def row2297 : SortedNatTree.Tree := (.node 11171 (.node 6269 (.node 4073 (.node 2531 (.node 2381 .empty .empty) (.node 3581 .empty .empty)) (.node 5081 (.node 4967 .empty .empty) (.node 6089 .empty .empty))) (.node 9293 (.node 7937 (.node 7127 .empty .empty) (.node 8807 .empty .empty)) (.node 10457 (.node 10037 .empty .empty) (.node 10781 .empty .empty)))) (.node 16649 (.node 12491 (.node 11927 (.node 11243 .empty .empty) (.node 12107 .empty .empty)) (.node 14153 (.node 12611 .empty .empty) (.node 15287 .empty .empty))) (.node 20639 (.node 19739 (.node 18257 .empty .empty) (.node 20549 .empty .empty)) (.node 23603 (.node 23447 .empty .empty) (.node 23633 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2297_ordered : SortedNatTree.ordered row2297 = true := by decide
set_option maxRecDepth 4096 in
theorem row2297_values : SortedNatTree.values row2297 = Data.row2297 := rfl
theorem row2297_checked : SortedNatTree.contains row2297 = Data.row2297.contains := by
  rw [SortedNatTree.contains_eq_list row2297 row2297_ordered, row2297_values]
def row2309 : SortedNatTree.Tree := (.node 12911 (.node 7691 (.node 5087 (.node 4133 (.node 3491 (.node 2687 .empty .empty) .empty) (.node 5081 (.node 4451 .empty .empty) .empty)) (.node 6131 (.node 5783 (.node 5417 .empty .empty) .empty) (.node 6947 (.node 6257 .empty .empty) .empty))) (.node 9887 (.node 8237 (.node 7829 (.node 7817 .empty .empty) .empty) (.node 8861 (.node 8681 .empty .empty) .empty)) (.node 11003 (.node 10391 (.node 10037 .empty .empty) .empty) (.node 12197 .empty .empty)))) (.node 19841 (.node 16547 (.node 14561 (.node 13313 (.node 13217 .empty .empty) .empty) (.node 15809 (.node 15551 .empty .empty) .empty)) (.node 18119 (.node 17573 (.node 16943 .empty .empty) .empty) (.node 19697 .empty .empty))) (.node 21683 (.node 21227 (.node 20369 (.node 19991 .empty .empty) .empty) (.node 21563 (.node 21419 .empty .empty) .empty)) (.node 23447 (.node 22817 (.node 22721 .empty .empty) .empty) (.node 23531 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2309_ordered : SortedNatTree.ordered row2309 = true := by decide
set_option maxRecDepth 4096 in
theorem row2309_values : SortedNatTree.values row2309 = Data.row2309 := rfl
theorem row2309_checked : SortedNatTree.contains row2309 = Data.row2309.contains := by
  rw [SortedNatTree.contains_eq_list row2309 row2309_ordered, row2309_values]
def row2311 : SortedNatTree.Tree := (.node 11833 (.node 6619 (.node 3943 (.node 3313 (.node 3061 (.node 2647 .empty .empty) .empty) (.node 3847 (.node 3529 .empty .empty) .empty)) (.node 6199 (.node 5179 (.node 4549 .empty .empty) .empty) (.node 6469 .empty .empty))) (.node 8941 (.node 7639 (.node 6883 (.node 6871 .empty .empty) .empty) (.node 8731 (.node 8287 .empty .empty) .empty)) (.node 10753 (.node 9739 (.node 9391 .empty .empty) .empty) (.node 11689 .empty .empty)))) (.node 17977 (.node 15277 (.node 14449 (.node 14029 (.node 13327 .empty .empty) .empty) (.node 15271 (.node 14731 .empty .empty) .empty)) (.node 17011 (.node 16981 (.node 16339 .empty .empty) .empty) (.node 17569 .empty .empty))) (.node 22063 (.node 21661 (.node 20023 (.node 19417 .empty .empty) .empty) (.node 21937 .empty .empty)) (.node 23593 (.node 22651 (.node 22273 .empty .empty) .empty) (.node 23689 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2311_ordered : SortedNatTree.ordered row2311 = true := by decide
set_option maxRecDepth 4096 in
theorem row2311_values : SortedNatTree.values row2311 = Data.row2311 := rfl
theorem row2311_checked : SortedNatTree.contains row2311 = Data.row2311.contains := by
  rw [SortedNatTree.contains_eq_list row2311 row2311_ordered, row2311_values]
def row2333 : SortedNatTree.Tree := (.node 12503 (.node 6311 (.node 3533 (.node 2963 (.node 2423 (.node 2339 .empty .empty) (.node 2843 .empty .empty)) (.node 3221 (.node 3137 .empty .empty) .empty)) (.node 5297 (.node 5189 (.node 3701 .empty .empty) (.node 5261 .empty .empty)) (.node 5987 (.node 5927 .empty .empty) .empty))) (.node 8831 (.node 7907 (.node 7229 (.node 6701 .empty .empty) (.node 7253 .empty .empty)) (.node 8429 (.node 8387 .empty .empty) .empty)) (.node 11489 (.node 9473 (.node 8867 .empty .empty) (.node 9479 .empty .empty)) (.node 12203 (.node 12119 .empty .empty) .empty)))) (.node 17837 (.node 15737 (.node 15101 (.node 14327 (.node 13109 .empty .empty) (.node 14867 .empty .empty)) (.node 15569 (.node 15413 .empty .empty) .empty)) (.node 16451 (.node 16253 (.node 15881 .empty .empty) (.node 16361 .empty .empty)) (.node 17627 (.node 16553 .empty .empty) .empty))) (.node 20231 (.node 19301 (.node 18143 (.node 18131 .empty .empty) (.node 19157 .empty .empty)) (.node 19949 (.node 19913 .empty .empty) .empty)) (.node 22751 (.node 20963 (.node 20753 .empty .empty) (.node 21701 .empty .empty)) (.node 23609 (.node 23399 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row2333_ordered : SortedNatTree.ordered row2333 = true := by decide
set_option maxRecDepth 4096 in
theorem row2333_values : SortedNatTree.values row2333 = Data.row2333 := rfl
theorem row2333_checked : SortedNatTree.contains row2333 = Data.row2333.contains := by
  rw [SortedNatTree.contains_eq_list row2333 row2333_ordered, row2333_values]
def row2339 : SortedNatTree.Tree := (.node 10037 (.node 7001 (.node 5039 (.node 3371 (.node 2729 (.node 2543 .empty .empty) (.node 3347 .empty .empty)) (.node 4337 (.node 4127 .empty .empty) .empty)) (.node 6047 (.node 5303 (.node 5081 .empty .empty) .empty) (.node 6287 (.node 6101 .empty .empty) .empty))) (.node 7841 (.node 7307 (.node 7247 (.node 7013 .empty .empty) .empty) (.node 7499 (.node 7487 .empty .empty) .empty)) (.node 9209 (.node 9011 (.node 8369 .empty .empty) .empty) (.node 9767 (.node 9743 .empty .empty) .empty)))) (.node 16451 (.node 14939 (.node 11927 (.node 11351 (.node 10709 .empty .empty) .empty) (.node 14741 (.node 12893 .empty .empty) .empty)) (.node 15959 (.node 15641 (.node 15467 .empty .empty) .empty) (.node 16253 (.node 16127 .empty .empty) .empty))) (.node 20807 (.node 18917 (.node 17903 (.node 17471 .empty .empty) .empty) (.node 20639 (.node 19319 .empty .empty) .empty)) (.node 22679 (.node 21503 (.node 21227 .empty .empty) .empty) (.node 23663 (.node 23603 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row2339_ordered : SortedNatTree.ordered row2339 = true := by decide
set_option maxRecDepth 4096 in
theorem row2339_values : SortedNatTree.values row2339 = Data.row2339 := rfl
theorem row2339_checked : SortedNatTree.contains row2339 = Data.row2339.contains := by
  rw [SortedNatTree.contains_eq_list row2339 row2339_ordered, row2339_values]
def row2341 : SortedNatTree.Tree := (.node 12379 (.node 8377 (.node 5107 (.node 3373 (.node 3259 (.node 2833 .empty .empty) .empty) (.node 3967 .empty .empty)) (.node 7879 (.node 7489 (.node 6571 .empty .empty) .empty) (.node 8179 .empty .empty))) (.node 10477 (.node 9859 (.node 9679 (.node 9511 .empty .empty) .empty) (.node 10099 .empty .empty)) (.node 11701 (.node 11083 .empty .empty) (.node 11953 .empty .empty)))) (.node 15907 (.node 14563 (.node 13309 (.node 13177 (.node 12409 .empty .empty) .empty) (.node 14479 .empty .empty)) (.node 15727 (.node 15439 .empty .empty) (.node 15901 .empty .empty))) (.node 21187 (.node 19441 (.node 19051 (.node 18433 .empty .empty) .empty) (.node 20353 .empty .empty)) (.node 22549 (.node 21313 .empty .empty) (.node 23557 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2341_ordered : SortedNatTree.ordered row2341 = true := by decide
set_option maxRecDepth 4096 in
theorem row2341_values : SortedNatTree.values row2341 = Data.row2341 := rfl
theorem row2341_checked : SortedNatTree.contains row2341 = Data.row2341.contains := by
  rw [SortedNatTree.contains_eq_list row2341 row2341_ordered, row2341_values]
def row2347 : SortedNatTree.Tree := (.node 9871 (.node 6469 (.node 3067 (.node 2833 (.node 2749 (.node 2731 .empty .empty) .empty) (.node 3049 (.node 2851 .empty .empty) .empty)) (.node 4639 (.node 4327 (.node 3253 .empty .empty) .empty) (.node 6199 (.node 4663 .empty .empty) .empty))) (.node 7879 (.node 7417 (.node 7057 (.node 6883 .empty .empty) .empty) (.node 7873 (.node 7681 .empty .empty) .empty)) (.node 9769 (.node 9151 (.node 8821 .empty .empty) .empty) (.node 9829 .empty .empty)))) (.node 16561 (.node 13693 (.node 12277 (.node 11311 (.node 10429 .empty .empty) .empty) (.node 12409 (.node 12343 .empty .empty) .empty)) (.node 15289 (.node 14713 (.node 14533 .empty .empty) .empty) (.node 15643 .empty .empty))) (.node 21247 (.node 19753 (.node 19717 (.node 19249 .empty .empty) .empty) (.node 20749 (.node 19819 .empty .empty) .empty)) (.node 22717 (.node 22273 (.node 22003 .empty .empty) .empty) (.node 22861 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2347_ordered : SortedNatTree.ordered row2347 = true := by decide
set_option maxRecDepth 4096 in
theorem row2347_values : SortedNatTree.values row2347 = Data.row2347 := rfl
theorem row2347_checked : SortedNatTree.contains row2347 = Data.row2347.contains := by
  rw [SortedNatTree.contains_eq_list row2347 row2347_ordered, row2347_values]
def row2351 : SortedNatTree.Tree := (.node 11171 (.node 7643 (.node 5519 (.node 3881 (.node 3863 (.node 3617 .empty .empty) .empty) (.node 5507 (.node 4973 .empty .empty) .empty)) (.node 7019 (.node 6959 (.node 6761 .empty .empty) .empty) (.node 7229 .empty .empty))) (.node 9059 (.node 8237 (.node 7853 (.node 7841 .empty .empty) .empty) (.node 8609 .empty .empty)) (.node 10331 (.node 9491 (.node 9341 .empty .empty) .empty) (.node 10529 .empty .empty)))) (.node 16421 (.node 13649 (.node 12503 (.node 12197 (.node 11621 .empty .empty) .empty) (.node 13397 .empty .empty)) (.node 14489 (.node 14423 (.node 13679 .empty .empty) .empty) (.node 14549 .empty .empty))) (.node 19853 (.node 19457 (.node 18287 (.node 17027 .empty .empty) .empty) (.node 19463 .empty .empty)) (.node 22571 (.node 21881 (.node 20147 .empty .empty) .empty) (.node 23117 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2351_ordered : SortedNatTree.ordered row2351 = true := by decide
set_option maxRecDepth 4096 in
theorem row2351_values : SortedNatTree.values row2351 = Data.row2351 := rfl
theorem row2351_checked : SortedNatTree.contains row2351 = Data.row2351.contains := by
  rw [SortedNatTree.contains_eq_list row2351 row2351_ordered, row2351_values]
def row2357 : SortedNatTree.Tree := (.node 8237 (.node 5711 (.node 3821 (.node 3467 (.node 2903 (.node 2579 .empty .empty) .empty) (.node 3527 .empty .empty)) (.node 5087 (.node 4289 (.node 4091 .empty .empty) .empty) (.node 5693 .empty .empty))) (.node 7331 (.node 6869 (.node 6551 (.node 6323 .empty .empty) .empty) (.node 7187 .empty .empty)) (.node 7919 (.node 7901 (.node 7457 .empty .empty) .empty) (.node 7937 .empty .empty)))) (.node 13691 (.node 10613 (.node 9431 (.node 8699 (.node 8447 .empty .empty) .empty) (.node 9851 .empty .empty)) (.node 12227 (.node 11633 (.node 11369 .empty .empty) .empty) (.node 12329 .empty .empty))) (.node 19013 (.node 15959 (.node 15923 (.node 14657 .empty .empty) .empty) (.node 16661 .empty .empty)) (.node 22031 (.node 20981 (.node 20201 .empty .empty) .empty) (.node 22787 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2357_ordered : SortedNatTree.ordered row2357 = true := by decide
set_option maxRecDepth 4096 in
theorem row2357_values : SortedNatTree.values row2357 = Data.row2357 := rfl
theorem row2357_checked : SortedNatTree.contains row2357 = Data.row2357.contains := by
  rw [SortedNatTree.contains_eq_list row2357 row2357_ordered, row2357_values]
def row2371 : SortedNatTree.Tree := (.node 13099 (.node 8287 (.node 3643 (.node 3229 (.node 3169 (.node 2917 .empty .empty) .empty) (.node 3607 .empty .empty)) (.node 4423 (.node 4021 .empty .empty) (.node 7219 .empty .empty))) (.node 10141 (.node 9967 (.node 9901 (.node 8641 .empty .empty) .empty) (.node 10099 .empty .empty)) (.node 12637 (.node 11821 .empty .empty) (.node 12703 .empty .empty)))) (.node 19609 (.node 16567 (.node 15121 (.node 13999 (.node 13441 .empty .empty) .empty) (.node 16189 .empty .empty)) (.node 17911 (.node 17827 .empty .empty) (.node 18289 .empty .empty))) (.node 21817 (.node 20749 (.node 19777 .empty .empty) (.node 21019 .empty .empty)) (.node 23197 (.node 22153 .empty .empty) (.node 23473 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2371_ordered : SortedNatTree.ordered row2371 = true := by decide
set_option maxRecDepth 4096 in
theorem row2371_values : SortedNatTree.values row2371 = Data.row2371 := rfl
theorem row2371_checked : SortedNatTree.contains row2371 = Data.row2371.contains := by
  rw [SortedNatTree.contains_eq_list row2371 row2371_ordered, row2371_values]
def row2377 : SortedNatTree.Tree := (.node 10243 (.node 6073 (.node 3433 (.node 3067 (.node 2797 (.node 2557 .empty .empty) .empty) (.node 3319 .empty .empty)) (.node 4327 (.node 4159 .empty .empty) (.node 5743 .empty .empty))) (.node 8719 (.node 6553 (.node 6199 .empty .empty) (.node 7603 .empty .empty)) (.node 9733 (.node 9649 .empty .empty) (.node 9949 .empty .empty)))) (.node 17509 (.node 13963 (.node 12637 (.node 10909 (.node 10657 .empty .empty) .empty) (.node 13627 .empty .empty)) (.node 16249 (.node 14437 .empty .empty) (.node 16693 .empty .empty))) (.node 20521 (.node 18301 (.node 18181 .empty .empty) (.node 19477 .empty .empty)) (.node 21943 (.node 21061 .empty .empty) (.node 22171 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2377_ordered : SortedNatTree.ordered row2377 = true := by decide
set_option maxRecDepth 4096 in
theorem row2377_values : SortedNatTree.values row2377 = Data.row2377 := rfl
theorem row2377_checked : SortedNatTree.contains row2377 = Data.row2377.contains := by
  rw [SortedNatTree.contains_eq_list row2377 row2377_ordered, row2377_values]
def row2381 : SortedNatTree.Tree := (.node 9923 (.node 6857 (.node 5483 (.node 3539 (.node 3221 (.node 2447 .empty .empty) .empty) (.node 4073 .empty .empty)) (.node 5807 (.node 5639 (.node 5501 .empty .empty) .empty) (.node 6563 .empty .empty))) (.node 8423 (.node 8117 (.node 7853 (.node 7487 .empty .empty) .empty) (.node 8369 .empty .empty)) (.node 9041 (.node 8807 .empty .empty) (.node 9491 .empty .empty)))) (.node 13313 (.node 11519 (.node 10979 (.node 10781 (.node 10193 .empty .empty) .empty) (.node 11087 .empty .empty)) (.node 12071 (.node 11789 .empty .empty) (.node 13037 .empty .empty))) (.node 20681 (.node 15887 (.node 14753 (.node 14081 .empty .empty) .empty) (.node 19727 .empty .empty)) (.node 21893 (.node 21347 .empty .empty) (.node 22739 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2381_ordered : SortedNatTree.ordered row2381 = true := by decide
set_option maxRecDepth 4096 in
theorem row2381_values : SortedNatTree.values row2381 = Data.row2381 := rfl
theorem row2381_checked : SortedNatTree.contains row2381 = Data.row2381.contains := by
  rw [SortedNatTree.contains_eq_list row2381 row2381_ordered, row2381_values]
def row2383 : SortedNatTree.Tree := (.node 11251 (.node 6733 (.node 4639 (.node 4021 (.node 3583 (.node 2659 .empty .empty) .empty) (.node 4273 (.node 4099 .empty .empty) .empty)) (.node 5431 (.node 5113 (.node 5023 .empty .empty) .empty) (.node 5527 .empty .empty))) (.node 8431 (.node 7873 (.node 7573 (.node 7129 .empty .empty) .empty) (.node 8263 .empty .empty)) (.node 9187 (.node 8761 (.node 8599 .empty .empty) .empty) (.node 9973 .empty .empty)))) (.node 18043 (.node 13807 (.node 12757 (.node 11923 (.node 11779 .empty .empty) .empty) (.node 13681 (.node 13591 .empty .empty) .empty)) (.node 17341 (.node 16369 (.node 14437 .empty .empty) .empty) (.node 17623 .empty .empty))) (.node 21277 (.node 19759 (.node 19597 (.node 19219 .empty .empty) .empty) (.node 19777 .empty .empty)) (.node 21649 (.node 21493 (.node 21319 .empty .empty) .empty) (.node 21943 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2383_ordered : SortedNatTree.ordered row2383 = true := by decide
set_option maxRecDepth 4096 in
theorem row2383_values : SortedNatTree.values row2383 = Data.row2383 := rfl
theorem row2383_checked : SortedNatTree.contains row2383 = Data.row2383.contains := by
  rw [SortedNatTree.contains_eq_list row2383 row2383_ordered, row2383_values]
def row2389 : SortedNatTree.Tree := (.node 9697 (.node 6529 (.node 4957 (.node 2767 (.node 2521 (.node 2503 .empty .empty) .empty) (.node 4243 (.node 3733 .empty .empty) .empty)) (.node 5479 (.node 5167 (.node 5023 .empty .empty) .empty) (.node 6427 .empty .empty))) (.node 7477 (.node 6907 (.node 6871 (.node 6703 .empty .empty) .empty) (.node 7351 .empty .empty)) (.node 8803 (.node 8761 (.node 7561 .empty .empty) .empty) (.node 9619 .empty .empty)))) (.node 16063 (.node 12433 (.node 12007 (.node 11731 (.node 10513 .empty .empty) .empty) (.node 12301 .empty .empty)) (.node 15661 (.node 13309 (.node 12823 .empty .empty) .empty) (.node 15877 .empty .empty))) (.node 19231 (.node 17929 (.node 17659 (.node 16921 .empty .empty) .empty) (.node 18523 .empty .empty)) (.node 20707 (.node 20173 (.node 19381 .empty .empty) .empty) (.node 21937 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2389_ordered : SortedNatTree.ordered row2389 = true := by decide
set_option maxRecDepth 4096 in
theorem row2389_values : SortedNatTree.values row2389 = Data.row2389 := rfl
theorem row2389_checked : SortedNatTree.contains row2389 = Data.row2389.contains := by
  rw [SortedNatTree.contains_eq_list row2389 row2389_ordered, row2389_values]
def row2393 : SortedNatTree.Tree := (.node 10289 (.node 4919 (.node 3347 (.node 2663 (.node 2423 (.node 2411 .empty .empty) .empty) (.node 3083 (.node 2927 .empty .empty) .empty)) (.node 3821 (.node 3557 (.node 3491 .empty .empty) .empty) (.node 4241 (.node 3851 .empty .empty) .empty))) (.node 8069 (.node 6791 (.node 6329 (.node 5981 .empty .empty) .empty) (.node 7829 (.node 7127 .empty .empty) .empty)) (.node 9491 (.node 9371 (.node 8867 .empty .empty) .empty) (.node 10091 (.node 9551 .empty .empty) .empty)))) (.node 18119 (.node 14087 (.node 11519 (.node 11411 (.node 11003 .empty .empty) .empty) (.node 12713 (.node 11981 .empty .empty) .empty)) (.node 15803 (.node 14939 (.node 14249 .empty .empty) .empty) (.node 17387 (.node 16073 .empty .empty) .empty))) (.node 21881 (.node 20483 (.node 19961 (.node 18899 .empty .empty) .empty) (.node 21491 (.node 21179 .empty .empty) .empty)) (.node 23003 (.node 22901 (.node 22079 .empty .empty) .empty) (.node 23327 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2393_ordered : SortedNatTree.ordered row2393 = true := by decide
set_option maxRecDepth 4096 in
theorem row2393_values : SortedNatTree.values row2393 = Data.row2393 := rfl
theorem row2393_checked : SortedNatTree.contains row2393 = Data.row2393.contains := by
  rw [SortedNatTree.contains_eq_list row2393 row2393_ordered, row2393_values]
def row2399 : SortedNatTree.Tree := (.node 11807 (.node 6761 (.node 4457 (.node 3467 (.node 3251 (.node 2663 .empty .empty) .empty) (.node 3911 (.node 3821 .empty .empty) .empty)) (.node 6131 (.node 5717 (.node 4517 .empty .empty) .empty) (.node 6737 (.node 6701 .empty .empty) .empty))) (.node 8783 (.node 8081 (.node 7283 (.node 6977 .empty .empty) .empty) (.node 8363 (.node 8087 .empty .empty) .empty)) (.node 11369 (.node 10709 (.node 9623 .empty .empty) .empty) (.node 11801 (.node 11447 .empty .empty) .empty)))) (.node 15959 (.node 14639 (.node 14153 (.node 13799 (.node 13151 .empty .empty) .empty) (.node 14561 (.node 14447 .empty .empty) .empty)) (.node 15227 (.node 15077 (.node 14699 .empty .empty) .empty) (.node 15581 (.node 15233 .empty .empty) .empty))) (.node 21701 (.node 20123 (.node 19727 (.node 19553 .empty .empty) .empty) (.node 21587 (.node 20939 .empty .empty) .empty)) (.node 23339 (.node 22643 (.node 21863 .empty .empty) .empty) (.node 23609 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2399_ordered : SortedNatTree.ordered row2399 = true := by decide
set_option maxRecDepth 4096 in
theorem row2399_values : SortedNatTree.values row2399 = Data.row2399 := rfl
theorem row2399_checked : SortedNatTree.contains row2399 = Data.row2399.contains := by
  rw [SortedNatTree.contains_eq_list row2399 row2399_ordered, row2399_values]
def row2411 : SortedNatTree.Tree := (.node 10253 (.node 7433 (.node 4919 (.node 4253 (.node 4241 (.node 3917 .empty .empty) .empty) (.node 4793 .empty .empty)) (.node 5879 (.node 5519 (.node 5171 .empty .empty) .empty) (.node 7019 .empty .empty))) (.node 9281 (.node 8447 (.node 8273 (.node 7481 .empty .empty) .empty) (.node 9257 .empty .empty)) (.node 9533 (.node 9497 .empty .empty) (.node 9803 .empty .empty)))) (.node 17099 (.node 13043 (.node 12107 (.node 11777 (.node 11069 .empty .empty) .empty) (.node 12899 .empty .empty)) (.node 16217 (.node 15269 .empty .empty) (.node 16691 .empty .empty))) (.node 19661 (.node 18869 (.node 18749 (.node 18287 .empty .empty) .empty) (.node 19553 .empty .empty)) (.node 20921 (.node 20543 .empty .empty) (.node 21821 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2411_ordered : SortedNatTree.ordered row2411 = true := by decide
set_option maxRecDepth 4096 in
theorem row2411_values : SortedNatTree.values row2411 = Data.row2411 := rfl
theorem row2411_checked : SortedNatTree.contains row2411 = Data.row2411.contains := by
  rw [SortedNatTree.contains_eq_list row2411 row2411_ordered, row2411_values]
def row2417 : SortedNatTree.Tree := (.node 11159 (.node 6857 (.node 5303 (.node 4229 (.node 3449 (.node 2741 .empty .empty) (.node 4001 .empty .empty)) (.node 5153 (.node 4253 .empty .empty) .empty)) (.node 6389 (.node 5927 (.node 5903 .empty .empty) .empty) (.node 6599 (.node 6563 .empty .empty) .empty))) (.node 8747 (.node 7691 (.node 7103 (.node 7019 .empty .empty) .empty) (.node 8237 (.node 7877 .empty .empty) .empty)) (.node 9437 (.node 8951 (.node 8867 .empty .empty) .empty) (.node 10463 (.node 9461 .empty .empty) .empty)))) (.node 18143 (.node 13553 (.node 12377 (.node 12101 (.node 11801 .empty .empty) .empty) (.node 13523 (.node 13229 .empty .empty) .empty)) (.node 15647 (.node 14939 (.node 14489 .empty .empty) .empty) (.node 17579 (.node 17123 .empty .empty) .empty))) (.node 20693 (.node 18947 (.node 18341 (.node 18251 .empty .empty) .empty) (.node 19661 (.node 19427 .empty .empty) .empty)) (.node 22727 (.node 21617 (.node 20753 .empty .empty) .empty) (.node 23099 (.node 22973 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row2417_ordered : SortedNatTree.ordered row2417 = true := by decide
set_option maxRecDepth 4096 in
theorem row2417_values : SortedNatTree.values row2417 = Data.row2417 := rfl
theorem row2417_checked : SortedNatTree.contains row2417 = Data.row2417.contains := by
  rw [SortedNatTree.contains_eq_list row2417 row2417_ordered, row2417_values]
def row2423 : SortedNatTree.Tree := (.node 11117 (.node 7727 (.node 4547 (.node 3863 (.node 3851 (.node 3359 .empty .empty) .empty) (.node 4451 (.node 4391 .empty .empty) .empty)) (.node 5303 (.node 5147 (.node 5021 .empty .empty) .empty) (.node 5669 (.node 5639 .empty .empty) .empty))) (.node 9227 (.node 8609 (.node 8273 (.node 7919 .empty .empty) .empty) (.node 9221 (.node 9209 .empty .empty) .empty)) (.node 10193 (.node 10181 (.node 9857 .empty .empty) .empty) (.node 10847 .empty .empty)))) (.node 20219 (.node 15767 (.node 13913 (.node 13679 (.node 12113 .empty .empty) .empty) (.node 15443 (.node 14153 .empty .empty) .empty)) (.node 19079 (.node 18341 (.node 16433 .empty .empty) .empty) (.node 20063 (.node 19541 .empty .empty) .empty))) (.node 22109 (.node 21347 (.node 21221 (.node 20411 .empty .empty) .empty) (.node 21503 (.node 21407 .empty .empty) .empty)) (.node 23087 (.node 22343 (.node 22247 .empty .empty) .empty) (.node 23327 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2423_ordered : SortedNatTree.ordered row2423 = true := by decide
set_option maxRecDepth 4096 in
theorem row2423_values : SortedNatTree.values row2423 = Data.row2423 := rfl
theorem row2423_checked : SortedNatTree.contains row2423 = Data.row2423.contains := by
  rw [SortedNatTree.contains_eq_list row2423 row2423_ordered, row2423_values]
def row2437 : SortedNatTree.Tree := (.node 10501 (.node 6073 (.node 3931 (.node 3541 (.node 3361 (.node 3301 .empty .empty) .empty) (.node 3889 .empty .empty)) (.node 4969 (.node 4621 (.node 4111 .empty .empty) .empty) (.node 5077 .empty .empty))) (.node 7027 (.node 6607 (.node 6337 (.node 6301 .empty .empty) .empty) (.node 6967 .empty .empty)) (.node 8353 (.node 7039 .empty .empty) (.node 10429 .empty .empty)))) (.node 17599 (.node 15319 (.node 13597 (.node 12379 (.node 12277 .empty .empty) .empty) (.node 14011 .empty .empty)) (.node 16519 (.node 15733 .empty .empty) (.node 16741 .empty .empty))) (.node 19309 (.node 18169 (.node 18127 (.node 17863 .empty .empty) .empty) (.node 18451 .empty .empty)) (.node 22669 (.node 20929 .empty .empty) (.node 23251 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2437_ordered : SortedNatTree.ordered row2437 = true := by decide
set_option maxRecDepth 4096 in
theorem row2437_values : SortedNatTree.values row2437 = Data.row2437 := rfl
theorem row2437_checked : SortedNatTree.contains row2437 = Data.row2437.contains := by
  rw [SortedNatTree.contains_eq_list row2437 row2437_ordered, row2437_values]
def row2441 : SortedNatTree.Tree := (.node 10169 (.node 5381 (.node 4349 (.node 3671 (.node 3209 (.node 2543 .empty .empty) .empty) (.node 4157 (.node 3677 .empty .empty) .empty)) (.node 5147 (.node 5087 (.node 4787 .empty .empty) .empty) (.node 5333 (.node 5297 .empty .empty) .empty))) (.node 7691 (.node 6659 (.node 6113 (.node 6089 .empty .empty) .empty) (.node 7643 (.node 7121 .empty .empty) .empty)) (.node 8423 (.node 8231 (.node 7727 .empty .empty) .empty) (.node 9929 (.node 9341 .empty .empty) .empty)))) (.node 17207 (.node 15083 (.node 11423 (.node 11321 (.node 10457 .empty .empty) .empty) (.node 14867 (.node 14051 .empty .empty) .empty)) (.node 15413 (.node 15329 (.node 15149 .empty .empty) .empty) (.node 15959 (.node 15671 .empty .empty) .empty))) (.node 20477 (.node 17597 (.node 17387 (.node 17351 .empty .empty) .empty) (.node 19973 (.node 19181 .empty .empty) .empty)) (.node 21863 (.node 21023 (.node 20897 .empty .empty) .empty) (.node 23333 (.node 22709 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row2441_ordered : SortedNatTree.ordered row2441 = true := by decide
set_option maxRecDepth 4096 in
theorem row2441_values : SortedNatTree.values row2441 = Data.row2441 := rfl
theorem row2441_checked : SortedNatTree.contains row2441 = Data.row2441.contains := by
  rw [SortedNatTree.contains_eq_list row2441 row2441_ordered, row2441_values]
def row2447 : SortedNatTree.Tree := (.node 11351 (.node 7001 (.node 5813 (.node 4019 (.node 3797 (.node 3011 .empty .empty) .empty) (.node 4937 .empty .empty)) (.node 6113 (.node 5939 .empty .empty) (.node 6323 .empty .empty))) (.node 9293 (.node 8093 (.node 7907 (.node 7283 .empty .empty) .empty) (.node 8123 .empty .empty)) (.node 10259 (.node 10211 .empty .empty) (.node 10889 .empty .empty)))) (.node 18839 (.node 13697 (.node 12911 (.node 12659 (.node 12197 .empty .empty) .empty) (.node 13463 .empty .empty)) (.node 15971 (.node 14033 .empty .empty) (.node 16553 .empty .empty))) (.node 21017 (.node 20129 (.node 19841 .empty .empty) (.node 21011 .empty .empty)) (.node 23417 (.node 21179 .empty .empty) (.node 23537 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2447_ordered : SortedNatTree.ordered row2447 = true := by decide
set_option maxRecDepth 4096 in
theorem row2447_values : SortedNatTree.values row2447 = Data.row2447 := rfl
theorem row2447_checked : SortedNatTree.contains row2447 = Data.row2447.contains := by
  rw [SortedNatTree.contains_eq_list row2447 row2447_ordered, row2447_values]
def row2459 : SortedNatTree.Tree := (.node 11261 (.node 6899 (.node 4919 (.node 4139 (.node 2927 (.node 2657 .empty .empty) .empty) (.node 4463 (.node 4289 .empty .empty) .empty)) (.node 6101 (.node 5843 (.node 5171 .empty .empty) .empty) (.node 6353 .empty .empty))) (.node 9059 (.node 8069 (.node 8009 (.node 7211 .empty .empty) .empty) (.node 8807 (.node 8423 .empty .empty) .empty)) (.node 10859 (.node 10799 (.node 9161 .empty .empty) .empty) (.node 11087 .empty .empty)))) (.node 18059 (.node 14561 (.node 14153 (.node 13757 (.node 12143 .empty .empty) .empty) (.node 14177 (.node 14159 .empty .empty) .empty)) (.node 16217 (.node 15077 (.node 14897 .empty .empty) .empty) (.node 17807 .empty .empty))) (.node 21701 (.node 19073 (.node 18731 (.node 18593 .empty .empty) .empty) (.node 20231 .empty .empty)) (.node 23333 (.node 23189 (.node 21737 .empty .empty) .empty) (.node 23459 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2459_ordered : SortedNatTree.ordered row2459 = true := by decide
set_option maxRecDepth 4096 in
theorem row2459_values : SortedNatTree.values row2459 = Data.row2459 := rfl
theorem row2459_checked : SortedNatTree.contains row2459 = Data.row2459.contains := by
  rw [SortedNatTree.contains_eq_list row2459 row2459_ordered, row2459_values]
def row2467 : SortedNatTree.Tree := (.node 11329 (.node 6529 (.node 4861 (.node 3217 (.node 2551 (.node 2539 .empty .empty) .empty) (.node 4723 (.node 4051 .empty .empty) .empty)) (.node 5689 (.node 5683 (.node 5407 .empty .empty) .empty) (.node 6277 (.node 6037 .empty .empty) .empty))) (.node 9601 (.node 8167 (.node 7963 (.node 7621 .empty .empty) .empty) (.node 8923 (.node 8377 .empty .empty) .empty)) (.node 10861 (.node 9931 (.node 9769 .empty .empty) .empty) (.node 10957 .empty .empty)))) (.node 16417 (.node 14437 (.node 13477 (.node 12919 (.node 11821 .empty .empty) .empty) (.node 13921 (.node 13627 .empty .empty) .empty)) (.node 15349 (.node 14983 (.node 14653 .empty .empty) .empty) (.node 15823 .empty .empty))) (.node 22027 (.node 17659 (.node 17539 (.node 16741 .empty .empty) .empty) (.node 19927 (.node 19801 .empty .empty) .empty)) (.node 22741 (.node 22501 (.node 22447 .empty .empty) .empty) (.node 23371 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2467_ordered : SortedNatTree.ordered row2467 = true := by decide
set_option maxRecDepth 4096 in
theorem row2467_values : SortedNatTree.values row2467 = Data.row2467 := rfl
theorem row2467_checked : SortedNatTree.contains row2467 = Data.row2467.contains := by
  rw [SortedNatTree.contains_eq_list row2467 row2467_ordered, row2467_values]
def row2473 : SortedNatTree.Tree := (.node 14149 (.node 6379 (.node 3889 (.node 2791 (.node 2713 (.node 2647 .empty .empty) .empty) (.node 3343 (.node 3001 .empty .empty) .empty)) (.node 5437 (.node 5023 (.node 4327 .empty .empty) .empty) (.node 5881 .empty .empty))) (.node 11131 (.node 8269 (.node 7417 (.node 6427 .empty .empty) .empty) (.node 10957 (.node 8941 .empty .empty) .empty)) (.node 13249 (.node 12301 (.node 11689 .empty .empty) .empty) (.node 13903 .empty .empty)))) (.node 19381 (.node 16903 (.node 15373 (.node 15307 (.node 15217 .empty .empty) .empty) (.node 16417 (.node 15493 .empty .empty) .empty)) (.node 18367 (.node 17599 (.node 17359 .empty .empty) .empty) (.node 18541 .empty .empty))) (.node 20809 (.node 20113 (.node 19867 (.node 19777 .empty .empty) .empty) (.node 20509 .empty .empty)) (.node 21961 (.node 21433 (.node 21391 .empty .empty) .empty) (.node 22003 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2473_ordered : SortedNatTree.ordered row2473 = true := by decide
set_option maxRecDepth 4096 in
theorem row2473_values : SortedNatTree.values row2473 = Data.row2473 := rfl
theorem row2473_checked : SortedNatTree.contains row2473 = Data.row2473.contains := by
  rw [SortedNatTree.contains_eq_list row2473 row2473_ordered, row2473_values]
def row2477 : SortedNatTree.Tree := (.node 10601 (.node 7853 (.node 4073 (.node 2939 (.node 2801 (.node 2663 .empty .empty) .empty) (.node 3797 .empty .empty)) (.node 5381 (.node 5039 (.node 4133 .empty .empty) .empty) (.node 7703 .empty .empty))) (.node 8807 (.node 8681 (.node 8669 (.node 8513 .empty .empty) .empty) (.node 8783 .empty .empty)) (.node 9719 (.node 9437 (.node 9341 .empty .empty) .empty) (.node 9929 .empty .empty)))) (.node 14591 (.node 13217 (.node 12251 (.node 12071 (.node 11549 .empty .empty) .empty) (.node 12821 .empty .empty)) (.node 13751 (.node 13577 (.node 13229 .empty .empty) .empty) (.node 13859 .empty .empty))) (.node 17027 (.node 16607 (.node 15737 (.node 15527 .empty .empty) .empty) (.node 16871 .empty .empty)) (.node 18617 (.node 17891 .empty .empty) (.node 19013 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2477_ordered : SortedNatTree.ordered row2477 = true := by decide
set_option maxRecDepth 4096 in
theorem row2477_values : SortedNatTree.values row2477 = Data.row2477 := rfl
theorem row2477_checked : SortedNatTree.contains row2477 = Data.row2477.contains := by
  rw [SortedNatTree.contains_eq_list row2477 row2477_ordered, row2477_values]
def row2503 : SortedNatTree.Tree := (.node 12619 (.node 6763 (.node 5581 (.node 3559 (.node 3517 (.node 3391 .empty .empty) .empty) (.node 5281 (.node 4447 .empty .empty) .empty)) (.node 6079 (.node 5869 (.node 5623 .empty .empty) .empty) (.node 6421 (.node 6163 .empty .empty) .empty))) (.node 9439 (.node 7591 (.node 7393 (.node 6967 .empty .empty) .empty) (.node 9319 (.node 8719 .empty .empty) .empty)) (.node 10993 (.node 9613 (.node 9601 .empty .empty) .empty) (.node 11827 .empty .empty)))) (.node 19231 (.node 17167 (.node 14149 (.node 13291 (.node 12637 .empty .empty) .empty) (.node 15913 (.node 15817 .empty .empty) .empty)) (.node 18181 (.node 17923 (.node 17581 .empty .empty) .empty) (.node 18979 (.node 18379 .empty .empty) .empty))) (.node 22381 (.node 20533 (.node 20143 (.node 19777 .empty .empty) .empty) (.node 22129 (.node 21529 .empty .empty) .empty)) (.node 23131 (.node 22717 (.node 22543 .empty .empty) .empty) (.node 23251 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2503_ordered : SortedNatTree.ordered row2503 = true := by decide
set_option maxRecDepth 4096 in
theorem row2503_values : SortedNatTree.values row2503 = Data.row2503 := rfl
theorem row2503_checked : SortedNatTree.contains row2503 = Data.row2503.contains := by
  rw [SortedNatTree.contains_eq_list row2503 row2503_ordered, row2503_values]
def row2521 : SortedNatTree.Tree := (.node 10597 (.node 5557 (.node 4201 (.node 3697 (.node 3271 (.node 2797 .empty .empty) .empty) (.node 3739 .empty .empty)) (.node 4759 (.node 4363 .empty .empty) (.node 4999 .empty .empty))) (.node 8929 (.node 7561 (.node 6343 .empty .empty) (.node 8887 .empty .empty)) (.node 9907 (.node 9781 .empty .empty) (.node 10273 .empty .empty)))) (.node 15649 (.node 12163 (.node 11593 (.node 11467 (.node 11329 .empty .empty) .empty) (.node 12097 .empty .empty)) (.node 13933 (.node 12517 .empty .empty) (.node 13999 .empty .empty))) (.node 18913 (.node 16747 (.node 16573 .empty .empty) (.node 18787 .empty .empty)) (.node 19753 (.node 19081 .empty .empty) (.node 21757 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2521_ordered : SortedNatTree.ordered row2521 = true := by decide
set_option maxRecDepth 4096 in
theorem row2521_values : SortedNatTree.values row2521 = Data.row2521 := rfl
theorem row2521_checked : SortedNatTree.contains row2521 = Data.row2521.contains := by
  rw [SortedNatTree.contains_eq_list row2521 row2521_ordered, row2521_values]
def row2531 : SortedNatTree.Tree := (.node 11867 (.node 6659 (.node 4889 (.node 3539 (.node 2741 (.node 2549 .empty .empty) .empty) (.node 4643 (.node 3797 .empty .empty) .empty)) (.node 5309 (.node 5273 (.node 4967 .empty .empty) .empty) (.node 6329 (.node 5651 .empty .empty) .empty))) (.node 8753 (.node 8123 (.node 7883 (.node 6701 .empty .empty) .empty) (.node 8669 (.node 8171 .empty .empty) .empty)) (.node 9293 (.node 9221 (.node 9011 .empty .empty) .empty) (.node 11471 (.node 9689 .empty .empty) .empty)))) (.node 16427 (.node 13577 (.node 12329 (.node 12149 (.node 11981 .empty .empty) .empty) (.node 12611 (.node 12413 .empty .empty) .empty)) (.node 14753 (.node 14633 (.node 14249 .empty .empty) .empty) (.node 16187 (.node 15809 .empty .empty) .empty))) (.node 18401 (.node 16943 (.node 16823 (.node 16649 .empty .empty) .empty) (.node 17957 (.node 17099 .empty .empty) .empty)) (.node 21893 (.node 20963 (.node 20759 .empty .empty) .empty) (.node 23291 (.node 22481 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row2531_ordered : SortedNatTree.ordered row2531 = true := by decide
set_option maxRecDepth 4096 in
theorem row2531_values : SortedNatTree.values row2531 = Data.row2531 := rfl
theorem row2531_checked : SortedNatTree.contains row2531 = Data.row2531.contains := by
  rw [SortedNatTree.contains_eq_list row2531 row2531_ordered, row2531_values]
def row2539 : SortedNatTree.Tree := (.node 12601 (.node 5449 (.node 4339 (.node 3463 (.node 3229 (.node 3163 .empty .empty) .empty) (.node 4297 .empty .empty)) (.node 5119 (.node 4723 (.node 4441 .empty .empty) .empty) (.node 5179 .empty .empty))) (.node 8707 (.node 7159 (.node 6823 (.node 6421 .empty .empty) .empty) (.node 7951 .empty .empty)) (.node 12097 (.node 9817 (.node 9511 .empty .empty) .empty) (.node 12511 .empty .empty)))) (.node 19069 (.node 15877 (.node 15643 (.node 14629 (.node 13999 .empty .empty) .empty) (.node 15661 .empty .empty)) (.node 17341 (.node 17029 (.node 16603 .empty .empty) .empty) (.node 17509 .empty .empty))) (.node 22111 (.node 20071 (.node 19843 (.node 19717 .empty .empty) .empty) (.node 21433 .empty .empty)) (.node 23227 (.node 23029 (.node 23017 .empty .empty) .empty) (.node 23269 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2539_ordered : SortedNatTree.ordered row2539 = true := by decide
set_option maxRecDepth 4096 in
theorem row2539_values : SortedNatTree.values row2539 = Data.row2539 := rfl
theorem row2539_checked : SortedNatTree.contains row2539 = Data.row2539.contains := by
  rw [SortedNatTree.contains_eq_list row2539 row2539_ordered, row2539_values]
def row2543 : SortedNatTree.Tree := (.node 10163 (.node 5483 (.node 4523 (.node 3851 (.node 3371 (.node 2609 .empty .empty) (.node 3539 .empty .empty)) (.node 4349 (.node 4337 .empty .empty) .empty)) (.node 5153 (.node 5147 (.node 5051 .empty .empty) .empty) (.node 5441 (.node 5237 .empty .empty) .empty))) (.node 7649 (.node 6911 (.node 6011 (.node 5591 .empty .empty) (.node 6791 .empty .empty)) (.node 7541 (.node 7331 .empty .empty) .empty)) (.node 9521 (.node 9413 (.node 9059 .empty .empty) .empty) (.node 9851 (.node 9539 .empty .empty) .empty)))) (.node 14897 (.node 12893 (.node 10499 (.node 10313 (.node 10253 .empty .empty) (.node 10487 .empty .empty)) (.node 11579 (.node 11261 .empty .empty) .empty)) (.node 14153 (.node 13259 (.node 12917 .empty .empty) .empty) (.node 14831 (.node 14537 .empty .empty) .empty))) (.node 17519 (.node 16301 (.node 15329 (.node 15107 .empty .empty) (.node 16253 .empty .empty)) (.node 17291 (.node 17123 .empty .empty) .empty)) (.node 21713 (.node 19973 (.node 17891 .empty .empty) .empty) (.node 22787 (.node 22709 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row2543_ordered : SortedNatTree.ordered row2543 = true := by decide
set_option maxRecDepth 4096 in
theorem row2543_values : SortedNatTree.values row2543 = Data.row2543 := rfl
theorem row2543_checked : SortedNatTree.contains row2543 = Data.row2543.contains := by
  rw [SortedNatTree.contains_eq_list row2543 row2543_ordered, row2543_values]
def row2549 : SortedNatTree.Tree := (.node 9743 (.node 6761 (.node 4019 (.node 3329 (.node 3089 (.node 2729 .empty .empty) .empty) (.node 3947 (.node 3593 .empty .empty) .empty)) (.node 5471 (.node 4349 (.node 4241 .empty .empty) .empty) (.node 6323 (.node 5501 .empty .empty) .empty))) (.node 7949 (.node 7673 (.node 7211 (.node 6833 .empty .empty) .empty) (.node 7937 (.node 7793 .empty .empty) .empty)) (.node 8747 (.node 8093 (.node 8009 .empty .empty) .empty) (.node 9209 .empty .empty)))) (.node 15467 (.node 13397 (.node 10979 (.node 10949 (.node 10037 .empty .empty) .empty) (.node 12401 (.node 11897 .empty .empty) .empty)) (.node 14087 (.node 14081 (.node 13553 .empty .empty) .empty) (.node 14783 (.node 14717 .empty .empty) .empty))) (.node 19961 (.node 19001 (.node 18899 (.node 18089 .empty .empty) .empty) (.node 19739 (.node 19013 .empty .empty) .empty)) (.node 22469 (.node 21683 (.node 21011 .empty .empty) .empty) (.node 22901 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2549_ordered : SortedNatTree.ordered row2549 = true := by decide
set_option maxRecDepth 4096 in
theorem row2549_values : SortedNatTree.values row2549 = Data.row2549 := rfl
theorem row2549_checked : SortedNatTree.contains row2549 = Data.row2549.contains := by
  rw [SortedNatTree.contains_eq_list row2549 row2549_ordered, row2549_values]
def row2551 : SortedNatTree.Tree := (.node 12379 (.node 6793 (.node 4969 (.node 4219 (.node 3331 (.node 2887 .empty .empty) .empty) (.node 4813 (.node 4591 .empty .empty) .empty)) (.node 6037 (.node 5227 (.node 5059 .empty .empty) .empty) (.node 6163 .empty .empty))) (.node 9859 (.node 9181 (.node 8719 (.node 7537 .empty .empty) .empty) (.node 9631 .empty .empty)) (.node 12007 (.node 11953 (.node 11743 .empty .empty) .empty) (.node 12157 .empty .empty)))) (.node 17377 (.node 15307 (.node 13681 (.node 12799 (.node 12763 .empty .empty) .empty) (.node 15031 .empty .empty)) (.node 16333 (.node 16231 (.node 15901 .empty .empty) .empty) (.node 16921 .empty .empty))) (.node 20101 (.node 19087 (.node 18289 (.node 18049 .empty .empty) .empty) (.node 19219 .empty .empty)) (.node 22171 (.node 21991 (.node 20113 .empty .empty) .empty) (.node 22447 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2551_ordered : SortedNatTree.ordered row2551 = true := by decide
set_option maxRecDepth 4096 in
theorem row2551_values : SortedNatTree.values row2551 = Data.row2551 := rfl
theorem row2551_checked : SortedNatTree.contains row2551 = Data.row2551.contains := by
  rw [SortedNatTree.contains_eq_list row2551 row2551_ordered, row2551_values]
def row2557 : SortedNatTree.Tree := (.node 11173 (.node 6997 (.node 4657 (.node 4243 (.node 3313 (.node 2683 .empty .empty) .empty) (.node 4639 .empty .empty)) (.node 6763 (.node 5749 (.node 4957 .empty .empty) .empty) (.node 6871 .empty .empty))) (.node 8191 (.node 7789 (.node 7687 (.node 7369 .empty .empty) .empty) (.node 7873 .empty .empty)) (.node 8821 (.node 8521 .empty .empty) (.node 10009 .empty .empty)))) (.node 15877 (.node 12157 (.node 12049 (.node 11689 (.node 11527 .empty .empty) .empty) (.node 12097 .empty .empty)) (.node 13477 (.node 12973 .empty .empty) (.node 13873 .empty .empty))) (.node 19801 (.node 18313 (.node 16987 (.node 16843 .empty .empty) .empty) (.node 19483 .empty .empty)) (.node 22639 (.node 19963 .empty .empty) (.node 23311 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2557_ordered : SortedNatTree.ordered row2557 = true := by decide
set_option maxRecDepth 4096 in
theorem row2557_values : SortedNatTree.values row2557 = Data.row2557 := rfl
theorem row2557_checked : SortedNatTree.contains row2557 = Data.row2557.contains := by
  rw [SortedNatTree.contains_eq_list row2557 row2557_ordered, row2557_values]
def row2579 : SortedNatTree.Tree := (.node 11483 (.node 6131 (.node 4871 (.node 4007 (.node 3623 (.node 3359 .empty .empty) .empty) (.node 4421 .empty .empty)) (.node 5639 (.node 4931 .empty .empty) (.node 5843 .empty .empty))) (.node 11177 (.node 7727 (.node 7349 (.node 7229 .empty .empty) .empty) (.node 10559 .empty .empty)) (.node 11273 (.node 11261 .empty .empty) (.node 11393 .empty .empty)))) (.node 19463 (.node 13841 (.node 12113 (.node 11987 (.node 11783 .empty .empty) .empty) (.node 12527 .empty .empty)) (.node 17519 (.node 15173 .empty .empty) (.node 19373 .empty .empty))) (.node 22271 (.node 20759 (.node 19919 .empty .empty) (.node 20879 .empty .empty)) (.node 23027 (.node 22859 .empty .empty) (.node 23399 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2579_ordered : SortedNatTree.ordered row2579 = true := by decide
set_option maxRecDepth 4096 in
theorem row2579_values : SortedNatTree.values row2579 = Data.row2579 := rfl
theorem row2579_checked : SortedNatTree.contains row2579 = Data.row2579.contains := by
  rw [SortedNatTree.contains_eq_list row2579 row2579_ordered, row2579_values]
def row2591 : SortedNatTree.Tree := (.node 11939 (.node 6299 (.node 4283 (.node 3527 (.node 3257 (.node 2633 .empty .empty) .empty) (.node 4211 (.node 4073 .empty .empty) .empty)) (.node 5807 (.node 5741 (.node 4937 .empty .empty) .empty) (.node 6197 (.node 5849 .empty .empty) .empty))) (.node 9767 (.node 7937 (.node 7499 (.node 7193 .empty .empty) .empty) (.node 8423 (.node 8231 .empty .empty) .empty)) (.node 11519 (.node 10151 (.node 9941 .empty .empty) .empty) (.node 11927 .empty .empty)))) (.node 19433 (.node 15959 (.node 13457 (.node 13121 (.node 13037 .empty .empty) .empty) (.node 15359 (.node 14249 .empty .empty) .empty)) (.node 16829 (.node 16223 (.node 16061 .empty .empty) .empty) (.node 17597 .empty .empty))) (.node 21911 (.node 21179 (.node 20759 (.node 19571 .empty .empty) .empty) (.node 21893 (.node 21683 .empty .empty) .empty)) (.node 22613 (.node 22541 (.node 21977 .empty .empty) .empty) (.node 23297 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2591_ordered : SortedNatTree.ordered row2591 = true := by decide
set_option maxRecDepth 4096 in
theorem row2591_values : SortedNatTree.values row2591 = Data.row2591 := rfl
theorem row2591_checked : SortedNatTree.contains row2591 = Data.row2591.contains := by
  rw [SortedNatTree.contains_eq_list row2591 row2591_ordered, row2591_values]
def row2593 : SortedNatTree.Tree := (.node 12421 (.node 5443 (.node 3307 (.node 2887 (.node 2713 (.node 2617 .empty .empty) (.node 2719 .empty .empty)) (.node 3169 (.node 2971 .empty .empty) .empty)) (.node 5011 (.node 3709 (.node 3373 .empty .empty) (.node 3847 .empty .empty)) (.node 5281 (.node 5101 .empty .empty) .empty))) (.node 8293 (.node 7573 (.node 7027 (.node 5683 .empty .empty) (.node 7039 .empty .empty)) (.node 7927 (.node 7753 .empty .empty) .empty)) (.node 9721 (.node 9421 (.node 9181 .empty .empty) (.node 9547 .empty .empty)) (.node 10789 (.node 10111 .empty .empty) .empty)))) (.node 18013 (.node 15619 (.node 12967 (.node 12637 (.node 12541 .empty .empty) (.node 12841 .empty .empty)) (.node 14851 (.node 13591 .empty .empty) .empty)) (.node 17209 (.node 16699 (.node 16411 .empty .empty) (.node 16903 .empty .empty)) (.node 17293 (.node 17239 .empty .empty) .empty))) (.node 21613 (.node 19861 (.node 19441 (.node 18427 .empty .empty) (.node 19507 .empty .empty)) (.node 21523 (.node 20389 .empty .empty) .empty)) (.node 22291 (.node 22027 (.node 21943 .empty .empty) .empty) (.node 23293 (.node 23209 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row2593_ordered : SortedNatTree.ordered row2593 = true := by decide
set_option maxRecDepth 4096 in
theorem row2593_values : SortedNatTree.values row2593 = Data.row2593 := rfl
theorem row2593_checked : SortedNatTree.contains row2593 = Data.row2593.contains := by
  rw [SortedNatTree.contains_eq_list row2593 row2593_ordered, row2593_values]
def row2609 : SortedNatTree.Tree := (.node 13007 (.node 6569 (.node 4241 (.node 3491 (.node 3257 (.node 2777 .empty .empty) .empty) (.node 3803 .empty .empty)) (.node 5651 (.node 4973 (.node 4931 .empty .empty) .empty) (.node 6197 .empty .empty))) (.node 8273 (.node 7829 (.node 7451 (.node 6599 .empty .empty) .empty) (.node 8069 .empty .empty)) (.node 10313 (.node 10037 (.node 8297 .empty .empty) .empty) (.node 11117 .empty .empty)))) (.node 19073 (.node 15107 (.node 14387 (.node 14033 (.node 13613 .empty .empty) .empty) (.node 14891 .empty .empty)) (.node 17837 (.node 16829 (.node 16127 .empty .empty) .empty) (.node 18131 .empty .empty))) (.node 22157 (.node 21587 (.node 20873 (.node 19937 .empty .empty) .empty) (.node 21803 .empty .empty)) (.node 22973 (.node 22433 (.node 22247 .empty .empty) .empty) (.node 23099 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2609_ordered : SortedNatTree.ordered row2609 = true := by decide
set_option maxRecDepth 4096 in
theorem row2609_values : SortedNatTree.values row2609 = Data.row2609 := rfl
theorem row2609_checked : SortedNatTree.contains row2609 = Data.row2609.contains := by
  rw [SortedNatTree.contains_eq_list row2609 row2609_ordered, row2609_values]
def row2617 : SortedNatTree.Tree := (.node 11941 (.node 7213 (.node 4273 (.node 2857 (.node 2833 (.node 2803 .empty .empty) .empty) (.node 3571 (.node 3181 .empty .empty) .empty)) (.node 5689 (.node 4801 (.node 4567 .empty .empty) .empty) (.node 5839 .empty .empty))) (.node 9157 (.node 8161 (.node 7741 (.node 7507 .empty .empty) .empty) (.node 8641 (.node 8629 .empty .empty) .empty)) (.node 10357 (.node 9817 (.node 9661 .empty .empty) .empty) (.node 10651 .empty .empty)))) (.node 17707 (.node 16381 (.node 15583 (.node 14923 (.node 12163 .empty .empty) .empty) (.node 16339 (.node 16033 .empty .empty) .empty)) (.node 16993 (.node 16759 (.node 16519 .empty .empty) .empty) (.node 17041 .empty .empty))) (.node 20509 (.node 19309 (.node 18661 (.node 17791 .empty .empty) .empty) (.node 20359 (.node 19333 .empty .empty) .empty)) (.node 21487 (.node 20551 (.node 20521 .empty .empty) .empty) (.node 21601 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2617_ordered : SortedNatTree.ordered row2617 = true := by decide
set_option maxRecDepth 4096 in
theorem row2617_values : SortedNatTree.values row2617 = Data.row2617 := rfl
theorem row2617_checked : SortedNatTree.contains row2617 = Data.row2617.contains := by
  rw [SortedNatTree.contains_eq_list row2617 row2617_ordered, row2617_values]
def row2621 : SortedNatTree.Tree := (.node 13367 (.node 6977 (.node 6053 (.node 4583 (.node 3623 (.node 2789 .empty .empty) .empty) (.node 5477 (.node 4973 .empty .empty) .empty)) (.node 6317 (.node 6263 (.node 6101 .empty .empty) .empty) (.node 6521 .empty .empty))) (.node 11699 (.node 8363 (.node 7817 (.node 7151 .empty .empty) .empty) (.node 8861 (.node 8663 .empty .empty) .empty)) (.node 12569 (.node 12203 (.node 11813 .empty .empty) .empty) (.node 13121 .empty .empty)))) (.node 16301 (.node 14081 (.node 13613 (.node 13457 (.node 13421 .empty .empty) .empty) (.node 13781 (.node 13757 .empty .empty) .empty)) (.node 15647 (.node 15629 (.node 15467 .empty .empty) .empty) (.node 16097 .empty .empty))) (.node 21803 (.node 19373 (.node 18413 (.node 17987 .empty .empty) .empty) (.node 19469 (.node 19463 .empty .empty) .empty)) (.node 23117 (.node 22739 (.node 22397 .empty .empty) .empty) (.node 23201 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2621_ordered : SortedNatTree.ordered row2621 = true := by decide
set_option maxRecDepth 4096 in
theorem row2621_values : SortedNatTree.values row2621 = Data.row2621 := rfl
theorem row2621_checked : SortedNatTree.contains row2621 = Data.row2621.contains := by
  rw [SortedNatTree.contains_eq_list row2621 row2621_ordered, row2621_values]
def row2633 : SortedNatTree.Tree := (.node 12149 (.node 7079 (.node 5147 (.node 4337 (.node 3929 (.node 2699 .empty .empty) .empty) (.node 5051 .empty .empty)) (.node 5867 (.node 5807 (.node 5693 .empty .empty) .empty) (.node 6581 .empty .empty))) (.node 8999 (.node 7547 (.node 7517 (.node 7193 .empty .empty) .empty) (.node 8369 .empty .empty)) (.node 11423 (.node 10847 (.node 10613 .empty .empty) .empty) (.node 11549 .empty .empty)))) (.node 16253 (.node 13763 (.node 13577 (.node 13469 (.node 13313 .empty .empty) .empty) (.node 13679 .empty .empty)) (.node 15959 (.node 14771 (.node 13859 .empty .empty) .empty) (.node 16223 .empty .empty))) (.node 20051 (.node 19751 (.node 17027 (.node 16631 .empty .empty) .empty) (.node 19949 .empty .empty)) (.node 21713 (.node 20147 .empty .empty) (.node 23003 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2633_ordered : SortedNatTree.ordered row2633 = true := by decide
set_option maxRecDepth 4096 in
theorem row2633_values : SortedNatTree.values row2633 = Data.row2633 := rfl
theorem row2633_checked : SortedNatTree.contains row2633 = Data.row2633.contains := by
  rw [SortedNatTree.contains_eq_list row2633 row2633_ordered, row2633_values]
def row2647 : SortedNatTree.Tree := (.node 13411 (.node 8563 (.node 6271 (.node 4663 (.node 3793 (.node 2851 .empty .empty) (.node 4057 .empty .empty)) (.node 6091 (.node 5431 .empty .empty) .empty)) (.node 6673 (.node 6577 (.node 6553 .empty .empty) .empty) (.node 8161 (.node 7951 .empty .empty) .empty))) (.node 12391 (.node 11113 (.node 8971 (.node 8737 .empty .empty) (.node 9151 .empty .empty)) (.node 11731 (.node 11197 .empty .empty) .empty)) (.node 12739 (.node 12697 (.node 12583 .empty .empty) .empty) (.node 13219 (.node 12853 .empty .empty) .empty)))) (.node 17317 (.node 15361 (.node 14071 (.node 13789 (.node 13441 .empty .empty) (.node 13807 .empty .empty)) (.node 15289 (.node 15259 .empty .empty) .empty)) (.node 16057 (.node 15889 (.node 15511 .empty .empty) .empty) (.node 17293 (.node 16993 .empty .empty) .empty))) (.node 19387 (.node 18301 (.node 18211 (.node 17377 .empty .empty) .empty) (.node 19069 (.node 18397 .empty .empty) .empty)) (.node 21277 (.node 20359 (.node 19963 .empty .empty) .empty) (.node 21727 (.node 21673 .empty .empty) .empty)))))
set_option maxRecDepth 4096 in
theorem row2647_ordered : SortedNatTree.ordered row2647 = true := by decide
set_option maxRecDepth 4096 in
theorem row2647_values : SortedNatTree.values row2647 = Data.row2647 := rfl
theorem row2647_checked : SortedNatTree.contains row2647 = Data.row2647.contains := by
  rw [SortedNatTree.contains_eq_list row2647 row2647_ordered, row2647_values]
def row2657 : SortedNatTree.Tree := (.node 13619 (.node 7247 (.node 5099 (.node 4391 (.node 3923 (.node 3719 .empty .empty) .empty) (.node 4517 .empty .empty)) (.node 6449 (.node 5279 .empty .empty) (.node 6833 .empty .empty))) (.node 9587 (.node 8963 (.node 8243 .empty .empty) (.node 9221 .empty .empty)) (.node 12269 (.node 11867 .empty .empty) (.node 12653 .empty .empty)))) (.node 18917 (.node 17477 (.node 15773 (.node 14633 .empty .empty) (.node 16481 .empty .empty)) (.node 18443 (.node 18191 .empty .empty) (.node 18539 .empty .empty))) (.node 20921 (.node 19121 (.node 19031 .empty .empty) (.node 20639 .empty .empty)) (.node 22901 (.node 21143 .empty .empty) (.node 23333 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2657_ordered : SortedNatTree.ordered row2657 = true := by decide
set_option maxRecDepth 4096 in
theorem row2657_values : SortedNatTree.values row2657 = Data.row2657 := rfl
theorem row2657_checked : SortedNatTree.contains row2657 = Data.row2657.contains := by
  rw [SortedNatTree.contains_eq_list row2657 row2657_ordered, row2657_values]
def row2659 : SortedNatTree.Tree := (.node 10627 (.node 5167 (.node 3907 (.node 3631 (.node 3169 (.node 3079 .empty .empty) .empty) (.node 3739 .empty .empty)) (.node 4759 (.node 4723 (.node 4423 .empty .empty) .empty) (.node 4933 .empty .empty))) (.node 7603 (.node 5869 (.node 5857 (.node 5827 .empty .empty) .empty) (.node 6949 .empty .empty)) (.node 9091 (.node 8641 (.node 7927 .empty .empty) .empty) (.node 9127 .empty .empty)))) (.node 17737 (.node 12853 (.node 12109 (.node 11287 (.node 10837 .empty .empty) .empty) (.node 12379 .empty .empty)) (.node 16477 (.node 14503 (.node 12979 .empty .empty) .empty) (.node 17497 .empty .empty))) (.node 20959 (.node 20233 (.node 18973 (.node 18229 .empty .empty) .empty) (.node 20857 .empty .empty)) (.node 22543 (.node 21757 .empty .empty) (.node 22777 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2659_ordered : SortedNatTree.ordered row2659 = true := by decide
set_option maxRecDepth 4096 in
theorem row2659_values : SortedNatTree.values row2659 = Data.row2659 := rfl
theorem row2659_checked : SortedNatTree.contains row2659 = Data.row2659.contains := by
  rw [SortedNatTree.contains_eq_list row2659 row2659_ordered, row2659_values]
def row2663 : SortedNatTree.Tree := (.node 11489 (.node 5849 (.node 4457 (.node 3533 (.node 3251 (.node 3119 .empty .empty) .empty) (.node 3779 .empty .empty)) (.node 5669 (.node 5471 .empty .empty) (.node 5813 .empty .empty))) (.node 8783 (.node 7901 (.node 7607 (.node 6359 .empty .empty) .empty) (.node 8291 .empty .empty)) (.node 10007 (.node 9293 .empty .empty) (.node 10259 .empty .empty)))) (.node 16097 (.node 14369 (.node 13997 (.node 13337 (.node 12479 .empty .empty) .empty) (.node 14327 .empty .empty)) (.node 14957 (.node 14891 .empty .empty) (.node 15173 .empty .empty))) (.node 21521 (.node 19421 (.node 16943 (.node 16553 .empty .empty) .empty) (.node 21467 .empty .empty)) (.node 22853 (.node 21863 .empty .empty) (.node 23081 .empty .empty)))))
set_option maxRecDepth 4096 in
theorem row2663_ordered : SortedNatTree.ordered row2663 = true := by decide
set_option maxRecDepth 4096 in
theorem row2663_values : SortedNatTree.values row2663 = Data.row2663 := rfl
theorem row2663_checked : SortedNatTree.contains row2663 = Data.row2663.contains := by
  rw [SortedNatTree.contains_eq_list row2663 row2663_ordered, row2663_values]
end ProjectEulerSolutions.PrimePairLarge.FastRows
