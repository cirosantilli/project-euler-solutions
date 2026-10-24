import ProjectEulerSolutions.Lib.PrimePairLarge.Data
import ProjectEulerSolutions.Lib.SortedNatTree
namespace ProjectEulerSolutions.PrimePairLarge.FastRows
def row7583 : SortedNatTree.Tree := (.node 11927 (.node 10733 (.node 9521 (.node 8447 (.node 8093 .empty .empty) .empty) (.node 9689 .empty .empty)) (.node 11621 (.node 11369 (.node 11093 .empty .empty) .empty) (.node 11783 .empty .empty))) (.node 14387 (.node 14321 (.node 13613 (.node 13487 .empty .empty) .empty) (.node 14327 .empty .empty)) (.node 17477 (.node 15401 (.node 14561 .empty .empty) .empty) (.node 18089 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7583_ordered : SortedNatTree.ordered row7583 = true := by decide
set_option maxRecDepth 4096 in
theorem row7583_values : SortedNatTree.values row7583 = Data.row7583 := rfl
theorem row7583_checked : SortedNatTree.contains row7583 = Data.row7583.contains := by
  rw [SortedNatTree.contains_eq_list row7583 row7583_ordered, row7583_values]
def row7589 : SortedNatTree.Tree := (.node 12239 (.node 9743 (.node 9029 (.node 8969 (.node 7949 .empty .empty) .empty) (.node 9587 (.node 9221 .empty .empty) .empty)) (.node 10613 (.node 10343 (.node 10079 .empty .empty) .empty) (.node 11939 (.node 11393 .empty .empty) .empty))) (.node 15569 (.node 14423 (.node 13241 (.node 13163 .empty .empty) .empty) (.node 14771 (.node 14447 .empty .empty) .empty)) (.node 17669 (.node 17483 (.node 16187 .empty .empty) .empty) (.node 18257 (.node 18119 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row7589_ordered : SortedNatTree.ordered row7589 = true := by decide
set_option maxRecDepth 4096 in
theorem row7589_values : SortedNatTree.values row7589 = Data.row7589 := rfl
theorem row7589_checked : SortedNatTree.contains row7589 = Data.row7589.contains := by
  rw [SortedNatTree.contains_eq_list row7589 row7589_ordered, row7589_values]
def row7591 : SortedNatTree.Tree := (.node 12241 (.node 9631 (.node 8581 (.node 8431 (.node 7867 .empty .empty) .empty) (.node 9049 .empty .empty)) (.node 10909 (.node 10159 .empty .empty) (.node 11827 .empty .empty))) (.node 17383 (.node 15733 (.node 14347 .empty .empty) (.node 16903 .empty .empty)) (.node 17761 (.node 17449 .empty .empty) (.node 17791 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7591_ordered : SortedNatTree.ordered row7591 = true := by decide
set_option maxRecDepth 4096 in
theorem row7591_values : SortedNatTree.values row7591 = Data.row7591 := rfl
theorem row7591_checked : SortedNatTree.contains row7591 = Data.row7591.contains := by
  rw [SortedNatTree.contains_eq_list row7591 row7591_ordered, row7591_values]
def row7603 : SortedNatTree.Tree := (.node 14323 (.node 11173 (.node 9631 (.node 8971 (.node 8731 .empty .empty) .empty) (.node 10501 .empty .empty)) (.node 12841 (.node 12697 (.node 11971 .empty .empty) .empty) (.node 13597 .empty .empty))) (.node 15679 (.node 14869 (.node 14827 (.node 14767 .empty .empty) .empty) (.node 15289 .empty .empty)) (.node 16741 (.node 15733 .empty .empty) (.node 17863 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7603_ordered : SortedNatTree.ordered row7603 = true := by decide
set_option maxRecDepth 4096 in
theorem row7603_values : SortedNatTree.values row7603 = Data.row7603 := rfl
theorem row7603_checked : SortedNatTree.contains row7603 = Data.row7603.contains := by
  rw [SortedNatTree.contains_eq_list row7603 row7603_ordered, row7603_values]
def row7607 : SortedNatTree.Tree := (.node 15161 (.node 10739 (.node 7853 (.node 7649 .empty .empty) (.node 8741 .empty .empty)) (.node 13907 (.node 11411 .empty .empty) .empty)) (.node 17093 (.node 16067 (.node 15377 .empty .empty) (.node 16127 .empty .empty)) (.node 18191 (.node 18041 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row7607_ordered : SortedNatTree.ordered row7607 = true := by decide
set_option maxRecDepth 4096 in
theorem row7607_values : SortedNatTree.values row7607 = Data.row7607 := rfl
theorem row7607_checked : SortedNatTree.contains row7607 = Data.row7607.contains := by
  rw [SortedNatTree.contains_eq_list row7607 row7607_ordered, row7607_values]
def row7621 : SortedNatTree.Tree := (.node 12829 (.node 9739 (.node 9277 (.node 8269 (.node 7741 .empty .empty) .empty) (.node 9547 .empty .empty)) (.node 12511 (.node 12409 (.node 11149 .empty .empty) .empty) (.node 12517 .empty .empty))) (.node 14713 (.node 14281 (.node 13903 (.node 13537 .empty .empty) .empty) (.node 14449 .empty .empty)) (.node 15889 (.node 15091 .empty .empty) (.node 16633 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7621_ordered : SortedNatTree.ordered row7621 = true := by decide
set_option maxRecDepth 4096 in
theorem row7621_values : SortedNatTree.values row7621 = Data.row7621 := rfl
theorem row7621_checked : SortedNatTree.contains row7621 = Data.row7621.contains := by
  rw [SortedNatTree.contains_eq_list row7621 row7621_ordered, row7621_values]
def row7639 : SortedNatTree.Tree := (.node 12739 (.node 9907 (.node 9007 (.node 8689 (.node 8419 .empty .empty) .empty) (.node 9619 .empty .empty)) (.node 11923 (.node 11503 (.node 10723 .empty .empty) .empty) (.node 12511 .empty .empty))) (.node 17359 (.node 15913 (.node 15091 (.node 12979 .empty .empty) .empty) (.node 17137 .empty .empty)) (.node 17599 (.node 17383 .empty .empty) (.node 18289 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7639_ordered : SortedNatTree.ordered row7639 = true := by decide
set_option maxRecDepth 4096 in
theorem row7639_values : SortedNatTree.values row7639 = Data.row7639 := rfl
theorem row7639_checked : SortedNatTree.contains row7639 = Data.row7639.contains := by
  rw [SortedNatTree.contains_eq_list row7639 row7639_ordered, row7639_values]
def row7643 : SortedNatTree.Tree := (.node 12239 (.node 8837 (.node 8573 (.node 8363 (.node 7883 .empty .empty) .empty) (.node 8627 .empty .empty)) (.node 10343 (.node 9677 (.node 9227 .empty .empty) .empty) (.node 11987 .empty .empty))) (.node 13721 (.node 13331 (.node 12743 (.node 12323 .empty .empty) .empty) (.node 13679 .empty .empty)) (.node 15473 (.node 15077 (.node 14783 .empty .empty) .empty) (.node 16691 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7643_ordered : SortedNatTree.ordered row7643 = true := by decide
set_option maxRecDepth 4096 in
theorem row7643_values : SortedNatTree.values row7643 = Data.row7643 := rfl
theorem row7643_checked : SortedNatTree.contains row7643 = Data.row7643.contains := by
  rw [SortedNatTree.contains_eq_list row7643 row7643_ordered, row7643_values]
def row7649 : SortedNatTree.Tree := (.node 14537 (.node 10499 (.node 9539 (.node 9257 (.node 8423 .empty .empty) .empty) (.node 10169 .empty .empty)) (.node 12641 (.node 11831 (.node 11003 .empty .empty) .empty) (.node 13841 .empty .empty))) (.node 16253 (.node 15773 (.node 15401 (.node 15377 .empty .empty) .empty) (.node 15881 .empty .empty)) (.node 18059 (.node 17393 (.node 16619 .empty .empty) .empty) (.node 18269 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7649_ordered : SortedNatTree.ordered row7649 = true := by decide
set_option maxRecDepth 4096 in
theorem row7649_values : SortedNatTree.values row7649 = Data.row7649 := rfl
theorem row7649_checked : SortedNatTree.contains row7649 = Data.row7649.contains := by
  rw [SortedNatTree.contains_eq_list row7649 row7649_ordered, row7649_values]
def row7669 : SortedNatTree.Tree := (.node 12889 (.node 11491 (.node 9433 (.node 8803 (.node 8311 .empty .empty) (.node 8887 .empty .empty)) (.node 11239 (.node 11071 .empty .empty) (.node 11329 .empty .empty))) (.node 12547 (.node 11863 (.node 11833 .empty .empty) (.node 12343 .empty .empty)) (.node 12799 (.node 12589 .empty .empty) (.node 12829 .empty .empty)))) (.node 16417 (.node 13921 (.node 13597 (.node 13291 .empty .empty) (.node 13723 .empty .empty)) (.node 14887 (.node 14821 .empty .empty) (.node 15439 .empty .empty))) (.node 17761 (.node 17419 (.node 16963 .empty .empty) (.node 17707 .empty .empty)) (.node 18211 (.node 17929 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row7669_ordered : SortedNatTree.ordered row7669 = true := by decide
set_option maxRecDepth 4096 in
theorem row7669_values : SortedNatTree.values row7669 = Data.row7669 := rfl
theorem row7669_checked : SortedNatTree.contains row7669 = Data.row7669.contains := by
  rw [SortedNatTree.contains_eq_list row7669 row7669_ordered, row7669_values]
def row7673 : SortedNatTree.Tree := (.node 13187 (.node 10007 (.node 8819 (.node 7823 .empty .empty) (.node 9851 .empty .empty)) (.node 12071 (.node 11549 .empty .empty) (.node 13121 .empty .empty))) (.node 16127 (.node 14087 (.node 14081 .empty .empty) (.node 15359 .empty .empty)) (.node 17909 (.node 17027 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row7673_ordered : SortedNatTree.ordered row7673 = true := by decide
set_option maxRecDepth 4096 in
theorem row7673_values : SortedNatTree.values row7673 = Data.row7673 := rfl
theorem row7673_checked : SortedNatTree.contains row7673 = Data.row7673.contains := by
  rw [SortedNatTree.contains_eq_list row7673 row7673_ordered, row7673_values]
def row7681 : SortedNatTree.Tree := (.node 13249 (.node 11251 (.node 9739 (.node 9091 (.node 8719 .empty .empty) .empty) (.node 10627 .empty .empty)) (.node 12739 (.node 12073 (.node 11593 .empty .empty) .empty) (.node 13183 .empty .empty))) (.node 16231 (.node 15541 (.node 14779 (.node 13921 .empty .empty) .empty) (.node 15919 .empty .empty)) (.node 18181 (.node 17929 .empty .empty) (.node 18199 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7681_ordered : SortedNatTree.ordered row7681 = true := by decide
set_option maxRecDepth 4096 in
theorem row7681_values : SortedNatTree.values row7681 = Data.row7681 := rfl
theorem row7681_checked : SortedNatTree.contains row7681 = Data.row7681.contains := by
  rw [SortedNatTree.contains_eq_list row7681 row7681_ordered, row7681_values]
def row7687 : SortedNatTree.Tree := (.node 12757 (.node 10651 (.node 8821 (.node 8521 (.node 8311 .empty .empty) .empty) (.node 9199 .empty .empty)) (.node 11701 (.node 11617 (.node 11197 .empty .empty) .empty) (.node 12277 .empty .empty))) (.node 15823 (.node 14083 (.node 13681 (.node 13339 .empty .empty) .empty) (.node 14461 .empty .empty)) (.node 17581 (.node 17359 .empty .empty) (.node 17977 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7687_ordered : SortedNatTree.ordered row7687 = true := by decide
set_option maxRecDepth 4096 in
theorem row7687_values : SortedNatTree.values row7687 = Data.row7687 := rfl
theorem row7687_checked : SortedNatTree.contains row7687 = Data.row7687.contains := by
  rw [SortedNatTree.contains_eq_list row7687 row7687_ordered, row7687_values]
def row7691 : SortedNatTree.Tree := (.node 14537 (.node 12269 (.node 10037 (.node 9371 (.node 8147 .empty .empty) .empty) (.node 11261 .empty .empty)) (.node 13757 (.node 13049 .empty .empty) (.node 14369 .empty .empty))) (.node 16787 (.node 15731 (.node 15569 .empty .empty) (.node 16631 .empty .empty)) (.node 18059 (.node 17957 .empty .empty) (.node 18143 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7691_ordered : SortedNatTree.ordered row7691 = true := by decide
set_option maxRecDepth 4096 in
theorem row7691_values : SortedNatTree.values row7691 = Data.row7691 := rfl
theorem row7691_checked : SortedNatTree.contains row7691 = Data.row7691.contains := by
  rw [SortedNatTree.contains_eq_list row7691 row7691_ordered, row7691_values]
def row7699 : SortedNatTree.Tree := (.node 13171 (.node 10459 (.node 9787 (.node 9679 (.node 7873 .empty .empty) .empty) (.node 10111 .empty .empty)) (.node 12421 (.node 11833 (.node 11743 .empty .empty) .empty) (.node 13093 .empty .empty))) (.node 16141 (.node 14293 (.node 13789 (.node 13681 .empty .empty) .empty) (.node 16087 .empty .empty)) (.node 16981 (.node 16267 (.node 16183 .empty .empty) .empty) (.node 17737 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7699_ordered : SortedNatTree.ordered row7699 = true := by decide
set_option maxRecDepth 4096 in
theorem row7699_values : SortedNatTree.values row7699 = Data.row7699 := rfl
theorem row7699_checked : SortedNatTree.contains row7699 = Data.row7699.contains := by
  rw [SortedNatTree.contains_eq_list row7699 row7699_ordered, row7699_values]
def row7703 : SortedNatTree.Tree := (.node 14249 (.node 12107 (.node 10979 (.node 7853 (.node 7823 .empty .empty) (.node 10889 .empty .empty)) (.node 11783 (.node 11027 .empty .empty) .empty)) (.node 12821 (.node 12539 (.node 12329 .empty .empty) .empty) (.node 13337 (.node 12899 .empty .empty) .empty))) (.node 16139 (.node 15731 (.node 14717 (.node 14423 .empty .empty) (.node 15263 .empty .empty)) (.node 16103 (.node 15761 .empty .empty) .empty)) (.node 16823 (.node 16553 (.node 16361 .empty .empty) .empty) (.node 17579 (.node 17519 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row7703_ordered : SortedNatTree.ordered row7703 = true := by decide
set_option maxRecDepth 4096 in
theorem row7703_values : SortedNatTree.values row7703 = Data.row7703 := rfl
theorem row7703_checked : SortedNatTree.contains row7703 = Data.row7703.contains := by
  rw [SortedNatTree.contains_eq_list row7703 row7703_ordered, row7703_values]
def row7717 : SortedNatTree.Tree := (.node 11527 (.node 8719 (.node 8389 (.node 7963 .empty .empty) (.node 8467 .empty .empty)) (.node 9901 (.node 9103 .empty .empty) (.node 10729 .empty .empty))) (.node 13627 (.node 12547 (.node 12541 .empty .empty) (.node 13093 .empty .empty)) (.node 17449 (.node 14347 .empty .empty) (.node 18253 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7717_ordered : SortedNatTree.ordered row7717 = true := by decide
set_option maxRecDepth 4096 in
theorem row7717_values : SortedNatTree.values row7717 = Data.row7717 := rfl
theorem row7717_checked : SortedNatTree.contains row7717 = Data.row7717.contains := by
  rw [SortedNatTree.contains_eq_list row7717 row7717_ordered, row7717_values]
def row7723 : SortedNatTree.Tree := (.node 14461 (.node 13033 (.node 12577 (.node 12241 .empty .empty) .empty) (.node 13291 .empty .empty)) (.node 15601 (.node 14533 .empty .empty) (.node 17569 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row7723_ordered : SortedNatTree.ordered row7723 = true := by decide
set_option maxRecDepth 4096 in
theorem row7723_values : SortedNatTree.values row7723 = Data.row7723 := rfl
theorem row7723_checked : SortedNatTree.contains row7723 = Data.row7723.contains := by
  rw [SortedNatTree.contains_eq_list row7723 row7723_ordered, row7723_values]
def row7727 : SortedNatTree.Tree := (.node 10949 (.node 8501 (.node 8219 (.node 7853 .empty .empty) (.node 8429 .empty .empty)) (.node 10343 (.node 10091 .empty .empty) (.node 10847 .empty .empty))) (.node 14951 (.node 13367 (.node 12641 .empty .empty) (.node 14561 .empty .empty)) (.node 17987 (.node 15287 .empty .empty) (.node 18269 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7727_ordered : SortedNatTree.ordered row7727 = true := by decide
set_option maxRecDepth 4096 in
theorem row7727_values : SortedNatTree.values row7727 = Data.row7727 := rfl
theorem row7727_checked : SortedNatTree.contains row7727 = Data.row7727.contains := by
  rw [SortedNatTree.contains_eq_list row7727 row7727_ordered, row7727_values]
def row7741 : SortedNatTree.Tree := (.node 13831 (.node 12589 (.node 9883 (.node 9337 (.node 8431 .empty .empty) .empty) (.node 10903 (.node 10093 .empty .empty) .empty)) (.node 13147 (.node 12613 (.node 12601 .empty .empty) .empty) (.node 13597 .empty .empty))) (.node 15679 (.node 14983 (.node 14713 (.node 14683 .empty .empty) .empty) (.node 15583 (.node 15061 .empty .empty) .empty)) (.node 17107 (.node 16843 (.node 16033 .empty .empty) .empty) (.node 18217 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7741_ordered : SortedNatTree.ordered row7741 = true := by decide
set_option maxRecDepth 4096 in
theorem row7741_values : SortedNatTree.values row7741 = Data.row7741 := rfl
theorem row7741_checked : SortedNatTree.contains row7741 = Data.row7741.contains := by
  rw [SortedNatTree.contains_eq_list row7741 row7741_ordered, row7741_values]
def row7753 : SortedNatTree.Tree := (.node 13633 (.node 9661 (.node 9109 (.node 8233 (.node 7927 .empty .empty) .empty) (.node 9283 .empty .empty)) (.node 13219 (.node 12457 (.node 9871 .empty .empty) .empty) (.node 13291 .empty .empty))) (.node 16411 (.node 15439 (.node 14341 (.node 13933 .empty .empty) .empty) (.node 15643 .empty .empty)) (.node 18121 (.node 17509 .empty .empty) (.node 18217 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7753_ordered : SortedNatTree.ordered row7753 = true := by decide
set_option maxRecDepth 4096 in
theorem row7753_values : SortedNatTree.values row7753 = Data.row7753 := rfl
theorem row7753_checked : SortedNatTree.contains row7753 = Data.row7753.contains := by
  rw [SortedNatTree.contains_eq_list row7753 row7753_ordered, row7753_values]
def row7757 : SortedNatTree.Tree := (.node 12323 (.node 9209 (.node 8297 (.node 7883 (.node 7853 .empty .empty) .empty) (.node 8597 .empty .empty)) (.node 10193 (.node 9521 (.node 9419 .empty .empty) .empty) (.node 11939 .empty .empty))) (.node 16229 (.node 15359 (.node 15107 (.node 12401 .empty .empty) .empty) (.node 15581 .empty .empty)) (.node 17231 (.node 16547 .empty .empty) (.node 17627 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7757_ordered : SortedNatTree.ordered row7757 = true := by decide
set_option maxRecDepth 4096 in
theorem row7757_values : SortedNatTree.values row7757 = Data.row7757 := rfl
theorem row7757_checked : SortedNatTree.contains row7757 = Data.row7757.contains := by
  rw [SortedNatTree.contains_eq_list row7757 row7757_ordered, row7757_values]
def row7759 : SortedNatTree.Tree := (.node 13381 (.node 10867 (.node 8929 (.node 8707 (.node 8467 .empty .empty) .empty) (.node 9091 (.node 8971 .empty .empty) .empty)) (.node 11863 (.node 10939 (.node 10909 .empty .empty) .empty) (.node 12301 (.node 12157 .empty .empty) .empty))) (.node 16657 (.node 14431 (.node 14143 (.node 13687 .empty .empty) .empty) (.node 15733 (.node 14437 .empty .empty) .empty)) (.node 17239 (.node 17203 (.node 16747 .empty .empty) .empty) (.node 18043 (.node 17341 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row7759_ordered : SortedNatTree.ordered row7759 = true := by decide
set_option maxRecDepth 4096 in
theorem row7759_values : SortedNatTree.values row7759 = Data.row7759 := rfl
theorem row7759_checked : SortedNatTree.contains row7759 = Data.row7759.contains := by
  rw [SortedNatTree.contains_eq_list row7759 row7759_ordered, row7759_values]
def row7789 : SortedNatTree.Tree := (.node 13249 (.node 9337 (.node 8521 (.node 7993 (.node 7867 .empty .empty) .empty) (.node 9109 (.node 8713 .empty .empty) .empty)) (.node 9949 (.node 9859 (.node 9697 .empty .empty) .empty) (.node 12763 .empty .empty))) (.node 15391 (.node 15259 (.node 14281 (.node 13477 .empty .empty) .empty) (.node 15307 .empty .empty)) (.node 17443 (.node 16189 (.node 15661 .empty .empty) .empty) (.node 17569 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7789_ordered : SortedNatTree.ordered row7789 = true := by decide
set_option maxRecDepth 4096 in
theorem row7789_values : SortedNatTree.values row7789 = Data.row7789 := rfl
theorem row7789_checked : SortedNatTree.contains row7789 = Data.row7789.contains := by
  rw [SortedNatTree.contains_eq_list row7789 row7789_ordered, row7789_values]
def row7793 : SortedNatTree.Tree := (.node 12143 (.node 9923 (.node 8807 (.node 8009 (.node 7919 .empty .empty) (.node 8363 .empty .empty)) (.node 9461 (.node 9161 .empty .empty) .empty)) (.node 11213 (.node 10667 (.node 10607 .empty .empty) (.node 11003 .empty .empty)) (.node 11681 (.node 11411 .empty .empty) .empty))) (.node 14741 (.node 12539 (.node 12329 (.node 12269 .empty .empty) (.node 12377 .empty .empty)) (.node 12983 (.node 12917 .empty .empty) .empty)) (.node 16871 (.node 16301 (.node 15161 .empty .empty) (.node 16361 .empty .empty)) (.node 17807 (.node 17159 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row7793_ordered : SortedNatTree.ordered row7793 = true := by decide
set_option maxRecDepth 4096 in
theorem row7793_values : SortedNatTree.values row7793 = Data.row7793 := rfl
theorem row7793_checked : SortedNatTree.contains row7793 = Data.row7793.contains := by
  rw [SortedNatTree.contains_eq_list row7793 row7793_ordered, row7793_values]
def row7817 : SortedNatTree.Tree := (.node 12791 (.node 9833 (.node 9371 (.node 8861 (.node 8147 .empty .empty) .empty) (.node 9749 .empty .empty)) (.node 12641 (.node 11369 (.node 10859 .empty .empty) .empty) (.node 12713 .empty .empty))) (.node 15227 (.node 14657 (.node 14411 (.node 12923 .empty .empty) .empty) (.node 14831 .empty .empty)) (.node 16673 (.node 15923 (.node 15731 .empty .empty) .empty) (.node 17393 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7817_ordered : SortedNatTree.ordered row7817 = true := by decide
set_option maxRecDepth 4096 in
theorem row7817_values : SortedNatTree.values row7817 = Data.row7817 := rfl
theorem row7817_checked : SortedNatTree.contains row7817 = Data.row7817.contains := by
  rw [SortedNatTree.contains_eq_list row7817 row7817_ordered, row7817_values]
def row7823 : SortedNatTree.Tree := (.node 13151 (.node 12281 (.node 10733 (.node 10007 (.node 8093 .empty .empty) .empty) (.node 11807 .empty .empty)) (.node 12917 (.node 12437 .empty .empty) (.node 12959 .empty .empty))) (.node 15971 (.node 15551 (.node 15467 (.node 13457 .empty .empty) .empty) (.node 15923 .empty .empty)) (.node 17231 (.node 16103 .empty .empty) (.node 17903 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7823_ordered : SortedNatTree.ordered row7823 = true := by decide
set_option maxRecDepth 4096 in
theorem row7823_values : SortedNatTree.values row7823 = Data.row7823 := rfl
theorem row7823_checked : SortedNatTree.contains row7823 = Data.row7823.contains := by
  rw [SortedNatTree.contains_eq_list row7823 row7823_ordered, row7823_values]
def row7829 : SortedNatTree.Tree := (.node 14723 (.node 12107 (.node 11471 (.node 8837 .empty .empty) (.node 12071 .empty .empty)) (.node 14423 (.node 12227 .empty .empty) (.node 14621 .empty .empty))) (.node 16763 (.node 15473 (.node 15017 .empty .empty) (.node 16193 .empty .empty)) (.node 17957 (.node 17489 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row7829_ordered : SortedNatTree.ordered row7829 = true := by decide
set_option maxRecDepth 4096 in
theorem row7829_values : SortedNatTree.values row7829 = Data.row7829 := rfl
theorem row7829_checked : SortedNatTree.contains row7829 = Data.row7829.contains := by
  rw [SortedNatTree.contains_eq_list row7829 row7829_ordered, row7829_values]
def row7841 : SortedNatTree.Tree := (.node 14519 (.node 12821 (.node 12149 (.node 10433 .empty .empty) (.node 12713 .empty .empty)) (.node 13859 (.node 12893 .empty .empty) (.node 14057 .empty .empty))) (.node 16547 (.node 15749 (.node 15641 .empty .empty) (.node 15887 .empty .empty)) (.node 17837 (.node 16901 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row7841_ordered : SortedNatTree.ordered row7841 = true := by decide
set_option maxRecDepth 4096 in
theorem row7841_values : SortedNatTree.values row7841 = Data.row7841 := rfl
theorem row7841_checked : SortedNatTree.contains row7841 = Data.row7841.contains := by
  rw [SortedNatTree.contains_eq_list row7841 row7841_ordered, row7841_values]
def row7853 : SortedNatTree.Tree := (.node 13367 (.node 12041 (.node 11411 (.node 7883 .empty .empty) (.node 11831 .empty .empty)) (.node 12641 (.node 12113 .empty .empty) .empty)) (.node 14753 (.node 14657 (.node 13469 .empty .empty) .empty) (.node 16229 (.node 15683 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row7853_ordered : SortedNatTree.ordered row7853 = true := by decide
set_option maxRecDepth 4096 in
theorem row7853_values : SortedNatTree.values row7853 = Data.row7853 := rfl
theorem row7853_checked : SortedNatTree.contains row7853 = Data.row7853.contains := by
  rw [SortedNatTree.contains_eq_list row7853 row7853_ordered, row7853_values]
def row7867 : SortedNatTree.Tree := (.node 12379 (.node 9397 (.node 8779 (.node 8443 .empty .empty) (.node 9187 .empty .empty)) (.node 10321 (.node 10039 .empty .empty) (.node 11287 .empty .empty))) (.node 15877 (.node 14923 (.node 13831 .empty .empty) (.node 15373 .empty .empty)) (.node 16699 (.node 15991 .empty .empty) (.node 16759 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7867_ordered : SortedNatTree.ordered row7867 = true := by decide
set_option maxRecDepth 4096 in
theorem row7867_values : SortedNatTree.values row7867 = Data.row7867 := rfl
theorem row7867_checked : SortedNatTree.contains row7867 = Data.row7867.contains := by
  rw [SortedNatTree.contains_eq_list row7867 row7867_ordered, row7867_values]
def row7873 : SortedNatTree.Tree := (.node 12451 (.node 9547 (.node 8941 (.node 8353 (.node 8293 .empty .empty) .empty) (.node 9049 .empty .empty)) (.node 10891 (.node 10831 (.node 10687 .empty .empty) .empty) (.node 11731 .empty .empty))) (.node 16603 (.node 14551 (.node 13729 (.node 13711 .empty .empty) .empty) (.node 15013 .empty .empty)) (.node 17599 (.node 16879 .empty .empty) (.node 17851 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7873_ordered : SortedNatTree.ordered row7873 = true := by decide
set_option maxRecDepth 4096 in
theorem row7873_values : SortedNatTree.values row7873 = Data.row7873 := rfl
theorem row7873_checked : SortedNatTree.contains row7873 = Data.row7873.contains := by
  rw [SortedNatTree.contains_eq_list row7873 row7873_ordered, row7873_values]
def row7877 : SortedNatTree.Tree := (.node 11867 (.node 9839 (.node 9227 (.node 9059 (.node 8597 .empty .empty) .empty) (.node 9461 (.node 9413 .empty .empty) .empty)) (.node 10289 (.node 10061 (.node 9923 .empty .empty) .empty) (.node 10667 .empty .empty))) (.node 14561 (.node 13151 (.node 13127 (.node 12479 .empty .empty) .empty) (.node 13691 .empty .empty)) (.node 16703 (.node 16091 (.node 16073 .empty .empty) .empty) (.node 17327 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7877_ordered : SortedNatTree.ordered row7877 = true := by decide
set_option maxRecDepth 4096 in
theorem row7877_values : SortedNatTree.values row7877 = Data.row7877 := rfl
theorem row7877_checked : SortedNatTree.contains row7877 = Data.row7877.contains := by
  rw [SortedNatTree.contains_eq_list row7877 row7877_ordered, row7877_values]
def row7879 : SortedNatTree.Tree := (.node 11959 (.node 9601 (.node 9157 (.node 9109 (.node 8353 .empty .empty) .empty) (.node 9439 (.node 9277 .empty .empty) .empty)) (.node 10939 (.node 10567 (.node 10267 .empty .empty) .empty) (.node 11617 (.node 11329 .empty .empty) .empty))) (.node 16231 (.node 12589 (.node 12547 (.node 12253 .empty .empty) .empty) (.node 16069 (.node 15319 .empty .empty) .empty)) (.node 17167 (.node 16693 (.node 16249 .empty .empty) .empty) (.node 17959 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7879_ordered : SortedNatTree.ordered row7879 = true := by decide
set_option maxRecDepth 4096 in
theorem row7879_values : SortedNatTree.values row7879 = Data.row7879 := rfl
theorem row7879_checked : SortedNatTree.contains row7879 = Data.row7879.contains := by
  rw [SortedNatTree.contains_eq_list row7879 row7879_ordered, row7879_values]
def row7883 : SortedNatTree.Tree := (.node 13217 (.node 9587 (.node 9473 (.node 8807 .empty .empty) (.node 9491 .empty .empty)) (.node 11399 (.node 11351 .empty .empty) (.node 11789 .empty .empty))) (.node 15761 (.node 14897 (.node 14783 .empty .empty) (.node 15413 .empty .empty)) (.node 16937 (.node 16547 .empty .empty) (.node 17909 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7883_ordered : SortedNatTree.ordered row7883 = true := by decide
set_option maxRecDepth 4096 in
theorem row7883_values : SortedNatTree.values row7883 = Data.row7883 := rfl
theorem row7883_checked : SortedNatTree.contains row7883 = Data.row7883.contains := by
  rw [SortedNatTree.contains_eq_list row7883 row7883_ordered, row7883_values]
def row7901 : SortedNatTree.Tree := (.node 15017 (.node 9539 (.node 8447 (.node 8081 .empty .empty) (.node 8741 .empty .empty)) (.node 11807 (.node 11027 .empty .empty) (.node 12161 .empty .empty))) (.node 17417 (.node 16427 (.node 16091 .empty .empty) (.node 16931 .empty .empty)) (.node 17957 (.node 17747 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row7901_ordered : SortedNatTree.ordered row7901 = true := by decide
set_option maxRecDepth 4096 in
theorem row7901_values : SortedNatTree.values row7901 = Data.row7901 := rfl
theorem row7901_checked : SortedNatTree.contains row7901 = Data.row7901.contains := by
  rw [SortedNatTree.contains_eq_list row7901 row7901_ordered, row7901_values]
def row7907 : SortedNatTree.Tree := (.node 12539 (.node 11027 (.node 8081 (.node 8039 (.node 7919 .empty .empty) .empty) (.node 10889 (.node 8297 .empty .empty) .empty)) (.node 12107 (.node 11807 (.node 11681 .empty .empty) .empty) (.node 12119 .empty .empty))) (.node 14843 (.node 13241 (.node 13127 (.node 12821 .empty .empty) .empty) (.node 14549 .empty .empty)) (.node 16607 (.node 16319 (.node 15377 .empty .empty) .empty) (.node 17117 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7907_ordered : SortedNatTree.ordered row7907 = true := by decide
set_option maxRecDepth 4096 in
theorem row7907_values : SortedNatTree.values row7907 = Data.row7907 := rfl
theorem row7907_checked : SortedNatTree.contains row7907 = Data.row7907.contains := by
  rw [SortedNatTree.contains_eq_list row7907 row7907_ordered, row7907_values]
def row7919 : SortedNatTree.Tree := (.node 12119 (.node 10883 (.node 9221 (.node 8117 .empty .empty) (.node 10529 .empty .empty)) (.node 11933 (.node 11801 .empty .empty) .empty)) (.node 14153 (.node 13679 (.node 12503 .empty .empty) .empty) (.node 17099 (.node 16361 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row7919_ordered : SortedNatTree.ordered row7919 = true := by decide
set_option maxRecDepth 4096 in
theorem row7919_values : SortedNatTree.values row7919 = Data.row7919 := rfl
theorem row7919_checked : SortedNatTree.contains row7919 = Data.row7919.contains := by
  rw [SortedNatTree.contains_eq_list row7919 row7919_ordered, row7919_values]
def row7927 : SortedNatTree.Tree := (.node 13309 (.node 12157 (.node 9463 (.node 9133 .empty .empty) (.node 9817 .empty .empty)) (.node 12301 (.node 12241 .empty .empty) (.node 12721 .empty .empty))) (.node 15199 (.node 14347 (.node 14107 .empty .empty) (.node 14779 .empty .empty)) (.node 16657 (.node 15241 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row7927_ordered : SortedNatTree.ordered row7927 = true := by decide
set_option maxRecDepth 4096 in
theorem row7927_values : SortedNatTree.values row7927 = Data.row7927 := rfl
theorem row7927_checked : SortedNatTree.contains row7927 = Data.row7927.contains := by
  rw [SortedNatTree.contains_eq_list row7927 row7927_ordered, row7927_values]
def row7933 : SortedNatTree.Tree := (.node 12781 (.node 10957 (.node 9643 (.node 9613 (.node 9133 .empty .empty) .empty) (.node 10513 .empty .empty)) (.node 12601 (.node 12517 .empty .empty) (.node 12697 .empty .empty))) (.node 14281 (.node 12889 (.node 12853 .empty .empty) (.node 13903 .empty .empty)) (.node 17443 (.node 16603 .empty .empty) (.node 17977 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7933_ordered : SortedNatTree.ordered row7933 = true := by decide
set_option maxRecDepth 4096 in
theorem row7933_values : SortedNatTree.values row7933 = Data.row7933 := rfl
theorem row7933_checked : SortedNatTree.contains row7933 = Data.row7933.contains := by
  rw [SortedNatTree.contains_eq_list row7933 row7933_ordered, row7933_values]
def row7937 : SortedNatTree.Tree := (.node 12653 (.node 10847 (.node 9413 (.node 8807 (.node 8081 .empty .empty) .empty) (.node 10091 .empty .empty)) (.node 12041 (.node 11777 (.node 11057 .empty .empty) .empty) (.node 12239 .empty .empty))) (.node 15467 (.node 13523 (.node 13241 (.node 13103 .empty .empty) .empty) (.node 13859 .empty .empty)) (.node 16433 (.node 16073 (.node 15473 .empty .empty) .empty) (.node 17387 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7937_ordered : SortedNatTree.ordered row7937 = true := by decide
set_option maxRecDepth 4096 in
theorem row7937_values : SortedNatTree.values row7937 = Data.row7937 := rfl
theorem row7937_checked : SortedNatTree.contains row7937 = Data.row7937.contains := by
  rw [SortedNatTree.contains_eq_list row7937 row7937_ordered, row7937_values]
def row7949 : SortedNatTree.Tree := (.node 11699 (.node 9437 (.node 8369 (.node 8297 (.node 8219 .empty .empty) .empty) (.node 8501 .empty .empty)) (.node 11093 (.node 10253 (.node 9923 .empty .empty) .empty) (.node 11597 .empty .empty))) (.node 15383 (.node 13337 (.node 13001 (.node 12689 .empty .empty) .empty) (.node 15137 .empty .empty)) (.node 16061 (.node 15773 (.node 15683 .empty .empty) .empty) (.node 16943 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7949_ordered : SortedNatTree.ordered row7949 = true := by decide
set_option maxRecDepth 4096 in
theorem row7949_values : SortedNatTree.values row7949 = Data.row7949 := rfl
theorem row7949_checked : SortedNatTree.contains row7949 = Data.row7949.contains := by
  rw [SortedNatTree.contains_eq_list row7949 row7949_ordered, row7949_values]
def row7951 : SortedNatTree.Tree := (.node 14713 (.node 9463 (.node 8779 (.node 8641 (.node 8623 .empty .empty) .empty) (.node 9439 (.node 9001 .empty .empty) .empty)) (.node 12907 (.node 11257 (.node 11113 .empty .empty) .empty) (.node 14197 .empty .empty))) (.node 16411 (.node 16111 (.node 15889 (.node 15679 .empty .empty) .empty) (.node 16363 .empty .empty)) (.node 17599 (.node 17293 (.node 16573 .empty .empty) .empty) (.node 17989 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7951_ordered : SortedNatTree.ordered row7951 = true := by decide
set_option maxRecDepth 4096 in
theorem row7951_values : SortedNatTree.values row7951 = Data.row7951 := rfl
theorem row7951_checked : SortedNatTree.contains row7951 = Data.row7951.contains := by
  rw [SortedNatTree.contains_eq_list row7951 row7951_ordered, row7951_values]
def row7963 : SortedNatTree.Tree := (.node 12457 (.node 10111 (.node 9187 (.node 9181 (.node 8011 .empty .empty) .empty) (.node 9199 .empty .empty)) (.node 11443 (.node 11239 (.node 10453 .empty .empty) .empty) (.node 11587 .empty .empty))) (.node 15391 (.node 14551 (.node 14011 (.node 12967 .empty .empty) .empty) (.node 14593 .empty .empty)) (.node 17377 (.node 15601 (.node 15439 .empty .empty) .empty) (.node 17881 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row7963_ordered : SortedNatTree.ordered row7963 = true := by decide
set_option maxRecDepth 4096 in
theorem row7963_values : SortedNatTree.values row7963 = Data.row7963 := rfl
theorem row7963_checked : SortedNatTree.contains row7963 = Data.row7963.contains := by
  rw [SortedNatTree.contains_eq_list row7963 row7963_ordered, row7963_values]
def row7993 : SortedNatTree.Tree := (.node 13003 (.node 10321 (.node 9199 (.node 9133 .empty .empty) (.node 9439 .empty .empty)) (.node 11239 (.node 10891 .empty .empty) .empty)) (.node 17317 (.node 16831 (.node 13441 .empty .empty) (.node 17209 .empty .empty)) (.node 17881 (.node 17851 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row7993_ordered : SortedNatTree.ordered row7993 = true := by decide
set_option maxRecDepth 4096 in
theorem row7993_values : SortedNatTree.values row7993 = Data.row7993 := rfl
theorem row7993_checked : SortedNatTree.contains row7993 = Data.row7993.contains := by
  rw [SortedNatTree.contains_eq_list row7993 row7993_ordered, row7993_values]
def row8009 : SortedNatTree.Tree := (.node 14969 (.node 10739 (.node 9419 (.node 8147 (.node 8093 .empty .empty) (.node 9311 .empty .empty)) (.node 10193 (.node 9491 .empty .empty) .empty)) (.node 13187 (.node 11621 (.node 11393 .empty .empty) (.node 12821 .empty .empty)) (.node 14249 (.node 13649 .empty .empty) .empty))) (.node 16661 (.node 15749 (.node 15263 (.node 15233 .empty .empty) (.node 15581 .empty .empty)) (.node 16619 (.node 16349 .empty .empty) .empty)) (.node 17189 (.node 17123 (.node 16703 .empty .empty) .empty) (.node 17921 (.node 17483 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row8009_ordered : SortedNatTree.ordered row8009 = true := by decide
set_option maxRecDepth 4096 in
theorem row8009_values : SortedNatTree.values row8009 = Data.row8009 := rfl
theorem row8009_checked : SortedNatTree.contains row8009 = Data.row8009.contains := by
  rw [SortedNatTree.contains_eq_list row8009 row8009_ordered, row8009_values]
def row8011 : SortedNatTree.Tree := (.node 12457 (.node 9439 (.node 8971 (.node 8707 (.node 8581 .empty .empty) .empty) (.node 9397 .empty .empty)) (.node 11821 (.node 11743 .empty .empty) (.node 12373 .empty .empty))) (.node 16831 (.node 15451 (.node 13711 (.node 12619 .empty .empty) .empty) (.node 16411 .empty .empty)) (.node 17467 (.node 17167 .empty .empty) (.node 17659 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8011_ordered : SortedNatTree.ordered row8011 = true := by decide
set_option maxRecDepth 4096 in
theorem row8011_values : SortedNatTree.values row8011 = Data.row8011 := rfl
theorem row8011_checked : SortedNatTree.contains row8011 = Data.row8011.contains := by
  rw [SortedNatTree.contains_eq_list row8011 row8011_ordered, row8011_values]
def row8017 : SortedNatTree.Tree := (.node 11527 (.node 10093 (.node 9349 (.node 9241 (.node 8779 .empty .empty) .empty) (.node 9613 (.node 9511 .empty .empty) .empty)) (.node 10957 (.node 10753 (.node 10627 .empty .empty) .empty) (.node 11059 .empty .empty))) (.node 15307 (.node 12517 (.node 11971 (.node 11839 .empty .empty) .empty) (.node 14503 (.node 13267 .empty .empty) .empty)) (.node 16963 (.node 16417 (.node 15973 .empty .empty) .empty) (.node 17827 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8017_ordered : SortedNatTree.ordered row8017 = true := by decide
set_option maxRecDepth 4096 in
theorem row8017_values : SortedNatTree.values row8017 = Data.row8017 := rfl
theorem row8017_checked : SortedNatTree.contains row8017 = Data.row8017.contains := by
  rw [SortedNatTree.contains_eq_list row8017 row8017_ordered, row8017_values]
def row8039 : SortedNatTree.Tree := (.node 14087 (.node 11699 (.node 10271 (.node 10181 (.node 8819 .empty .empty) .empty) (.node 11483 .empty .empty)) (.node 12647 (.node 12437 .empty .empty) (.node 12911 .empty .empty))) (.node 16319 (.node 15881 (.node 15383 (.node 14633 .empty .empty) .empty) (.node 16097 .empty .empty)) (.node 16901 (.node 16481 .empty .empty) (.node 17159 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8039_ordered : SortedNatTree.ordered row8039 = true := by decide
set_option maxRecDepth 4096 in
theorem row8039_values : SortedNatTree.values row8039 = Data.row8039 := rfl
theorem row8039_checked : SortedNatTree.contains row8039 = Data.row8039.contains := by
  rw [SortedNatTree.contains_eq_list row8039 row8039_ordered, row8039_values]
def row8053 : SortedNatTree.Tree := (.node 12589 (.node 10939 (.node 9103 (.node 8431 (.node 8287 .empty .empty) .empty) (.node 10861 .empty .empty)) (.node 11587 (.node 11491 (.node 10993 .empty .empty) .empty) (.node 11779 .empty .empty))) (.node 15733 (.node 14149 (.node 13591 (.node 12757 .empty .empty) .empty) (.node 14347 .empty .empty)) (.node 17209 (.node 15991 .empty .empty) (.node 17707 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8053_ordered : SortedNatTree.ordered row8053 = true := by decide
set_option maxRecDepth 4096 in
theorem row8053_values : SortedNatTree.values row8053 = Data.row8053 := rfl
theorem row8053_checked : SortedNatTree.contains row8053 = Data.row8053.contains := by
  rw [SortedNatTree.contains_eq_list row8053 row8053_ordered, row8053_values]
def row8059 : SortedNatTree.Tree := (.node 13249 (.node 9151 (.node 8527 (.node 8461 (.node 8269 .empty .empty) .empty) (.node 9067 (.node 8737 .empty .empty) .empty)) (.node 11863 (.node 10243 (.node 10093 .empty .empty) .empty) (.node 13159 (.node 12553 .empty .empty) .empty))) (.node 15601 (.node 15031 (.node 14323 (.node 13327 .empty .empty) .empty) (.node 15541 (.node 15289 .empty .empty) .empty)) (.node 16927 (.node 16087 (.node 15727 .empty .empty) .empty) (.node 17449 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8059_ordered : SortedNatTree.ordered row8059 = true := by decide
set_option maxRecDepth 4096 in
theorem row8059_values : SortedNatTree.values row8059 = Data.row8059 := rfl
theorem row8059_checked : SortedNatTree.contains row8059 = Data.row8059.contains := by
  rw [SortedNatTree.contains_eq_list row8059 row8059_ordered, row8059_values]
def row8069 : SortedNatTree.Tree := (.node 13463 (.node 10211 (.node 9497 (.node 9491 (.node 8861 .empty .empty) .empty) (.node 9803 .empty .empty)) (.node 11117 (.node 11003 .empty .empty) (.node 13421 .empty .empty))) (.node 14447 (.node 14057 (.node 14033 (.node 13781 .empty .empty) .empty) (.node 14177 .empty .empty)) (.node 16547 (.node 15737 .empty .empty) (.node 17939 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8069_ordered : SortedNatTree.ordered row8069 = true := by decide
set_option maxRecDepth 4096 in
theorem row8069_values : SortedNatTree.values row8069 = Data.row8069 := rfl
theorem row8069_checked : SortedNatTree.contains row8069 = Data.row8069.contains := by
  rw [SortedNatTree.contains_eq_list row8069 row8069_ordered, row8069_values]
def row8081 : SortedNatTree.Tree := (.node 11027 (.node 9419 (.node 8753 (.node 8537 .empty .empty) (.node 9161 .empty .empty)) (.node 10559 (.node 10253 .empty .empty) (.node 10847 .empty .empty))) (.node 15797 (.node 15053 (.node 11867 .empty .empty) (.node 15299 .empty .empty)) (.node 16553 (.node 16487 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row8081_ordered : SortedNatTree.ordered row8081 = true := by decide
set_option maxRecDepth 4096 in
theorem row8081_values : SortedNatTree.values row8081 = Data.row8081 := rfl
theorem row8081_checked : SortedNatTree.contains row8081 = Data.row8081.contains := by
  rw [SortedNatTree.contains_eq_list row8081 row8081_ordered, row8081_values]
def row8087 : SortedNatTree.Tree := (.node 13859 (.node 9587 (.node 9413 (.node 8219 .empty .empty) .empty) (.node 13463 (.node 11807 .empty .empty) .empty)) (.node 14411 (.node 13907 (.node 13883 .empty .empty) .empty) (.node 17123 (.node 16193 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row8087_ordered : SortedNatTree.ordered row8087 = true := by decide
set_option maxRecDepth 4096 in
theorem row8087_values : SortedNatTree.values row8087 = Data.row8087 := rfl
theorem row8087_checked : SortedNatTree.contains row8087 = Data.row8087.contains := by
  rw [SortedNatTree.contains_eq_list row8087 row8087_ordered, row8087_values]
def row8089 : SortedNatTree.Tree := (.node 12757 (.node 10357 (.node 8887 (.node 8737 (.node 8443 .empty .empty) (.node 8821 .empty .empty)) (.node 9829 (.node 9319 .empty .empty) .empty)) (.node 11197 (.node 11113 (.node 10903 .empty .empty) .empty) (.node 12541 (.node 11503 .empty .empty) .empty))) (.node 15493 (.node 14731 (.node 14173 (.node 12799 .empty .empty) (.node 14479 .empty .empty)) (.node 15277 (.node 14983 .empty .empty) .empty)) (.node 16333 (.node 16033 (.node 15661 .empty .empty) .empty) (.node 17713 (.node 16651 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row8089_ordered : SortedNatTree.ordered row8089 = true := by decide
set_option maxRecDepth 4096 in
theorem row8089_values : SortedNatTree.values row8089 = Data.row8089 := rfl
theorem row8089_checked : SortedNatTree.contains row8089 = Data.row8089.contains := by
  rw [SortedNatTree.contains_eq_list row8089 row8089_ordered, row8089_values]
def row8093 : SortedNatTree.Tree := (.node 11939 (.node 11483 (.node 9137 (.node 8999 .empty .empty) (.node 10457 .empty .empty)) (.node 11813 (.node 11801 .empty .empty) .empty)) (.node 15401 (.node 14843 (.node 14423 .empty .empty) (.node 14951 .empty .empty)) (.node 15497 (.node 15473 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row8093_ordered : SortedNatTree.ordered row8093 = true := by decide
set_option maxRecDepth 4096 in
theorem row8093_values : SortedNatTree.values row8093 = Data.row8093 := rfl
theorem row8093_checked : SortedNatTree.contains row8093 = Data.row8093.contains := by
  rw [SortedNatTree.contains_eq_list row8093 row8093_ordered, row8093_values]
def row8101 : SortedNatTree.Tree := (.node 12451 (.node 9643 (.node 8467 (.node 8317 (.node 8233 .empty .empty) .empty) (.node 9349 (.node 9241 .empty .empty) .empty)) (.node 11173 (.node 10501 (.node 9721 .empty .empty) .empty) (.node 11719 (.node 11503 .empty .empty) .empty))) (.node 13873 (.node 13309 (.node 12889 (.node 12703 .empty .empty) .empty) (.node 13537 (.node 13441 .empty .empty) .empty)) (.node 15391 (.node 14827 (.node 14407 .empty .empty) .empty) (.node 16477 (.node 15583 .empty .empty) .empty))))
set_option maxRecDepth 4096 in
theorem row8101_ordered : SortedNatTree.ordered row8101 = true := by decide
set_option maxRecDepth 4096 in
theorem row8101_values : SortedNatTree.values row8101 = Data.row8101 := rfl
theorem row8101_checked : SortedNatTree.contains row8101 = Data.row8101.contains := by
  rw [SortedNatTree.contains_eq_list row8101 row8101_ordered, row8101_values]
def row8111 : SortedNatTree.Tree := (.node 13103 (.node 9221 (.node 8573 (.node 8291 (.node 8123 .empty .empty) .empty) (.node 8963 .empty .empty)) (.node 10151 (.node 9743 .empty .empty) (.node 12941 .empty .empty))) (.node 16253 (.node 15269 (.node 14639 .empty .empty) (.node 15497 .empty .empty)) (.node 17597 (.node 16547 .empty .empty) (.node 17789 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8111_ordered : SortedNatTree.ordered row8111 = true := by decide
set_option maxRecDepth 4096 in
theorem row8111_values : SortedNatTree.values row8111 = Data.row8111 := rfl
theorem row8111_checked : SortedNatTree.contains row8111 = Data.row8111.contains := by
  rw [SortedNatTree.contains_eq_list row8111 row8111_ordered, row8111_values]
def row8117 : SortedNatTree.Tree := (.node 15809 (.node 11987 (.node 11261 (.node 10607 .empty .empty) .empty) (.node 13997 .empty .empty)) (.node 17657 (.node 16937 .empty .empty) (.node 17783 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row8117_ordered : SortedNatTree.ordered row8117 = true := by decide
set_option maxRecDepth 4096 in
theorem row8117_values : SortedNatTree.values row8117 = Data.row8117 := rfl
theorem row8117_checked : SortedNatTree.contains row8117 = Data.row8117.contains := by
  rw [SortedNatTree.contains_eq_list row8117 row8117_ordered, row8117_values]
def row8123 : SortedNatTree.Tree := (.node 13757 (.node 11657 (.node 9533 (.node 9377 .empty .empty) (.node 11399 .empty .empty)) (.node 13037 (.node 12101 .empty .empty) .empty)) (.node 16061 (.node 15269 (.node 14159 .empty .empty) .empty) (.node 17327 (.node 16943 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row8123_ordered : SortedNatTree.ordered row8123 = true := by decide
set_option maxRecDepth 4096 in
theorem row8123_values : SortedNatTree.values row8123 = Data.row8123 := rfl
theorem row8123_checked : SortedNatTree.contains row8123 = Data.row8123.contains := by
  rw [SortedNatTree.contains_eq_list row8123 row8123_ordered, row8123_values]
def row8147 : SortedNatTree.Tree := (.node 12911 (.node 9533 (.node 8753 (.node 8597 (.node 8219 .empty .empty) .empty) (.node 9173 (.node 8963 .empty .empty) .empty)) (.node 11897 (.node 11273 (.node 10889 .empty .empty) .empty) (.node 12377 .empty .empty))) (.node 14867 (.node 14207 (.node 14057 (.node 13931 .empty .empty) .empty) (.node 14759 (.node 14753 .empty .empty) .empty)) (.node 17471 (.node 15581 (.node 15263 .empty .empty) .empty) (.node 17609 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8147_ordered : SortedNatTree.ordered row8147 = true := by decide
set_option maxRecDepth 4096 in
theorem row8147_values : SortedNatTree.values row8147 = Data.row8147 := rfl
theorem row8147_checked : SortedNatTree.contains row8147 = Data.row8147.contains := by
  rw [SortedNatTree.contains_eq_list row8147 row8147_ordered, row8147_values]
def row8161 : SortedNatTree.Tree := (.node 12163 (.node 9103 (.node 8839 (.node 8389 (.node 8233 .empty .empty) .empty) (.node 8923 (.node 8863 .empty .empty) .empty)) (.node 11833 (.node 10093 (.node 9463 .empty .empty) .empty) (.node 12109 .empty .empty))) (.node 15649 (.node 13219 (.node 13171 (.node 12613 .empty .empty) .empty) (.node 15541 (.node 14407 .empty .empty) .empty)) (.node 16759 (.node 16033 (.node 15823 .empty .empty) .empty) (.node 17839 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8161_ordered : SortedNatTree.ordered row8161 = true := by decide
set_option maxRecDepth 4096 in
theorem row8161_values : SortedNatTree.values row8161 = Data.row8161 := rfl
theorem row8161_checked : SortedNatTree.contains row8161 = Data.row8161.contains := by
  rw [SortedNatTree.contains_eq_list row8161 row8161_ordered, row8161_values]
def row8167 : SortedNatTree.Tree := (.node 12409 (.node 9439 (.node 8563 (.node 8263 (.node 8209 .empty .empty) .empty) (.node 9001 .empty .empty)) (.node 11719 (.node 11251 .empty .empty) (.node 12157 .empty .empty))) (.node 16903 (.node 14551 (.node 14401 (.node 12979 .empty .empty) .empty) (.node 15877 .empty .empty)) (.node 16981 (.node 16963 .empty .empty) (.node 17659 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8167_ordered : SortedNatTree.ordered row8167 = true := by decide
set_option maxRecDepth 4096 in
theorem row8167_values : SortedNatTree.values row8167 = Data.row8167 := rfl
theorem row8167_checked : SortedNatTree.contains row8167 = Data.row8167.contains := by
  rw [SortedNatTree.contains_eq_list row8167 row8167_ordered, row8167_values]
def row8171 : SortedNatTree.Tree := (.node 12269 (.node 10709 (.node 10589 (.node 8807 (.node 8537 .empty .empty) .empty) (.node 10631 .empty .empty)) (.node 12041 (.node 11597 .empty .empty) (.node 12143 .empty .empty))) (.node 14657 (.node 13421 (.node 12899 .empty .empty) (.node 14387 .empty .empty)) (.node 15737 (.node 15581 .empty .empty) (.node 17117 .empty .empty))))
set_option maxRecDepth 4096 in
theorem row8171_ordered : SortedNatTree.ordered row8171 = true := by decide
set_option maxRecDepth 4096 in
theorem row8171_values : SortedNatTree.values row8171 = Data.row8171 := rfl
theorem row8171_checked : SortedNatTree.contains row8171 = Data.row8171.contains := by
  rw [SortedNatTree.contains_eq_list row8171 row8171_ordered, row8171_values]
end ProjectEulerSolutions.PrimePairLarge.FastRows
