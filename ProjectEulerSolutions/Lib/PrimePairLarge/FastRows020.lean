import ProjectEulerSolutions.Lib.PrimePairLarge.Data
import ProjectEulerSolutions.Lib.SortedNatTree
namespace ProjectEulerSolutions.PrimePairLarge.FastRows
def row10487 : SortedNatTree.Tree := (.node 12893 (.node 11681 (.node 11621 (.node 11579 .empty .empty) .empty) (.node 11777 .empty .empty)) (.node 14033 (.node 13709 .empty .empty) (.node 15287 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10487_ordered : SortedNatTree.ordered row10487 = true := by decide
set_option maxRecDepth 4096 in
theorem row10487_values : SortedNatTree.values row10487 = Data.row10487 := rfl
theorem row10487_checked : SortedNatTree.contains row10487 = Data.row10487.contains := by
  rw [SortedNatTree.contains_eq_list row10487 row10487_ordered, row10487_values]
def row10499 : SortedNatTree.Tree := (.node 12107 (.node 10859 (.node 10847 .empty .empty) (.node 10889 .empty .empty)) (.node 12527 (.node 12377 .empty .empty) (.node 13049 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10499_ordered : SortedNatTree.ordered row10499 = true := by decide
set_option maxRecDepth 4096 in
theorem row10499_values : SortedNatTree.values row10499 = Data.row10499 := rfl
theorem row10499_checked : SortedNatTree.contains row10499 = Data.row10499.contains := by
  rw [SortedNatTree.contains_eq_list row10499 row10499_ordered, row10499_values]
def row10501 : SortedNatTree.Tree := (.node 12109 (.node 11527 (.node 10531 .empty .empty) (.node 11731 .empty .empty)) (.node 14713 (.node 13291 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row10501_ordered : SortedNatTree.ordered row10501 = true := by decide
set_option maxRecDepth 4096 in
theorem row10501_values : SortedNatTree.values row10501 = Data.row10501 := rfl
theorem row10501_checked : SortedNatTree.contains row10501 = Data.row10501.contains := by
  rw [SortedNatTree.contains_eq_list row10501 row10501_ordered, row10501_values]
def row10513 : SortedNatTree.Tree := (.node 13537 (.node 11953 (.node 11923 (.node 10903 .empty .empty) .empty) (.node 13411 (.node 12511 .empty .empty) .empty)) (.node 14947 (.node 14503 (.node 13999 .empty .empty) .empty) (.node 15451 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10513_ordered : SortedNatTree.ordered row10513 = true := by decide
set_option maxRecDepth 4096 in
theorem row10513_values : SortedNatTree.values row10513 = Data.row10513 := rfl
theorem row10513_checked : SortedNatTree.contains row10513 = Data.row10513.contains := by
  rw [SortedNatTree.contains_eq_list row10513 row10513_ordered, row10513_values]
def row10529 : SortedNatTree.Tree := (.node 13313 (.node 12473 (.node 11159 (.node 10883 .empty .empty) .empty) (.node 13187 (.node 12899 .empty .empty) .empty)) (.node 14537 (.node 14519 (.node 13331 .empty .empty) .empty) (.node 15443 (.node 14699 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row10529_ordered : SortedNatTree.ordered row10529 = true := by decide
set_option maxRecDepth 4096 in
theorem row10529_values : SortedNatTree.values row10529 = Data.row10529 := rfl
theorem row10529_checked : SortedNatTree.contains row10529 = Data.row10529.contains := by
  rw [SortedNatTree.contains_eq_list row10529 row10529_ordered, row10529_values]
def row10531 : SortedNatTree.Tree := (.node 12421 (.node 12163 (.node 11467 (.node 11083 .empty .empty) .empty) (.node 12211 .empty .empty)) (.node 14221 (.node 13381 .empty .empty) (.node 14281 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10531_ordered : SortedNatTree.ordered row10531 = true := by decide
set_option maxRecDepth 4096 in
theorem row10531_values : SortedNatTree.values row10531 = Data.row10531 := rfl
theorem row10531_checked : SortedNatTree.contains row10531 = Data.row10531.contains := by
  rw [SortedNatTree.contains_eq_list row10531 row10531_ordered, row10531_values]
def row10559 : SortedNatTree.Tree := (.node 12713 (.node 11789 (.node 11699 (.node 10733 .empty .empty) .empty) (.node 12239 (.node 11897 .empty .empty) .empty)) (.node 14009 (.node 13709 (.node 13259 .empty .empty) .empty) (.node 15173 (.node 14879 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row10559_ordered : SortedNatTree.ordered row10559 = true := by decide
set_option maxRecDepth 4096 in
theorem row10559_values : SortedNatTree.values row10559 = Data.row10559 := rfl
theorem row10559_checked : SortedNatTree.contains row10559 = Data.row10559.contains := by
  rw [SortedNatTree.contains_eq_list row10559 row10559_ordered, row10559_values]
def row10567 : SortedNatTree.Tree := (.node 14341 (.node 13159 (.node 12541 (.node 12289 .empty .empty) .empty) (.node 13759 .empty .empty)) (.node 14851 (.node 14479 .empty .empty) (.node 15091 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10567_ordered : SortedNatTree.ordered row10567 = true := by decide
set_option maxRecDepth 4096 in
theorem row10567_values : SortedNatTree.values row10567 = Data.row10567 := rfl
theorem row10567_checked : SortedNatTree.contains row10567 = Data.row10567.contains := by
  rw [SortedNatTree.contains_eq_list row10567 row10567_ordered, row10567_values]
def row10589 : SortedNatTree.Tree := (.node 15149 (.node 12611 .empty .empty) .empty)
set_option maxRecDepth 4096 in
theorem row10589_ordered : SortedNatTree.ordered row10589 = true := by decide
set_option maxRecDepth 4096 in
theorem row10589_values : SortedNatTree.values row10589 = Data.row10589 := rfl
theorem row10589_checked : SortedNatTree.contains row10589 = Data.row10589.contains := by
  rw [SortedNatTree.contains_eq_list row10589 row10589_ordered, row10589_values]
def row10597 : SortedNatTree.Tree := (.node 11311 (.node 11197 (.node 10711 .empty .empty) (.node 11239 .empty .empty)) (.node 11971 (.node 11383 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row10597_ordered : SortedNatTree.ordered row10597 = true := by decide
set_option maxRecDepth 4096 in
theorem row10597_values : SortedNatTree.values row10597 = Data.row10597 := rfl
theorem row10597_checked : SortedNatTree.contains row10597 = Data.row10597.contains := by
  rw [SortedNatTree.contains_eq_list row10597 row10597_ordered, row10597_values]
def row10601 : SortedNatTree.Tree := (.node 12917 (.node 11903 (.node 11399 .empty .empty) .empty) (.node 15383 .empty .empty))
set_option maxRecDepth 4096 in
theorem row10601_ordered : SortedNatTree.ordered row10601 = true := by decide
set_option maxRecDepth 4096 in
theorem row10601_values : SortedNatTree.values row10601 = Data.row10601 := rfl
theorem row10601_checked : SortedNatTree.contains row10601 = Data.row10601.contains := by
  rw [SortedNatTree.contains_eq_list row10601 row10601_ordered, row10601_values]
def row10607 : SortedNatTree.Tree := (.node 15101 (.node 14741 (.node 14087 .empty .empty) .empty) (.node 15137 .empty .empty))
set_option maxRecDepth 4096 in
theorem row10607_ordered : SortedNatTree.ordered row10607 = true := by decide
set_option maxRecDepth 4096 in
theorem row10607_values : SortedNatTree.values row10607 = Data.row10607 := rfl
theorem row10607_checked : SortedNatTree.contains row10607 = Data.row10607.contains := by
  rw [SortedNatTree.contains_eq_list row10607 row10607_ordered, row10607_values]
def row10613 : SortedNatTree.Tree := (.node 13397 (.node 12809 (.node 11807 .empty .empty) .empty) (.node 14747 (.node 14699 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row10613_ordered : SortedNatTree.ordered row10613 = true := by decide
set_option maxRecDepth 4096 in
theorem row10613_values : SortedNatTree.values row10613 = Data.row10613 := rfl
theorem row10613_checked : SortedNatTree.contains row10613 = Data.row10613.contains := by
  rw [SortedNatTree.contains_eq_list row10613 row10613_ordered, row10613_values]
def row10627 : SortedNatTree.Tree := (.node 14143 (.node 13627 (.node 13009 .empty .empty) (.node 13999 .empty .empty)) (.node 14923 (.node 14281 .empty .empty) (.node 15193 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10627_ordered : SortedNatTree.ordered row10627 = true := by decide
set_option maxRecDepth 4096 in
theorem row10627_values : SortedNatTree.values row10627 = Data.row10627 := rfl
theorem row10627_checked : SortedNatTree.contains row10627 = Data.row10627.contains := by
  rw [SortedNatTree.contains_eq_list row10627 row10627_ordered, row10627_values]
def row10631 : SortedNatTree.Tree := (.node 14207 (.node 11069 (.node 10847 .empty .empty) .empty) (.node 14561 .empty .empty))
set_option maxRecDepth 4096 in
theorem row10631_ordered : SortedNatTree.ordered row10631 = true := by decide
set_option maxRecDepth 4096 in
theorem row10631_values : SortedNatTree.values row10631 = Data.row10631 := rfl
theorem row10631_checked : SortedNatTree.contains row10631 = Data.row10631.contains := by
  rw [SortedNatTree.contains_eq_list row10631 row10631_ordered, row10631_values]
def row10639 : SortedNatTree.Tree := (.node 12517 (.node 11503 (.node 11491 (.node 10891 .empty .empty) .empty) (.node 12433 (.node 11587 .empty .empty) .empty)) (.node 14461 (.node 14431 (.node 12637 .empty .empty) .empty) (.node 15271 (.node 14827 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row10639_ordered : SortedNatTree.ordered row10639 = true := by decide
set_option maxRecDepth 4096 in
theorem row10639_values : SortedNatTree.values row10639 = Data.row10639 := rfl
theorem row10639_checked : SortedNatTree.contains row10639 = Data.row10639.contains := by
  rw [SortedNatTree.contains_eq_list row10639 row10639_ordered, row10639_values]
def row10651 : SortedNatTree.Tree := (.node 12613 (.node 11197 (.node 10939 .empty .empty) (.node 11689 .empty .empty)) (.node 13963 (.node 13249 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row10651_ordered : SortedNatTree.ordered row10651 = true := by decide
set_option maxRecDepth 4096 in
theorem row10651_values : SortedNatTree.values row10651 = Data.row10651 := rfl
theorem row10651_checked : SortedNatTree.contains row10651 = Data.row10651.contains := by
  rw [SortedNatTree.contains_eq_list row10651 row10651_ordered, row10651_values]
def row10657 : SortedNatTree.Tree := (.node 12919 (.node 11971 (.node 11677 (.node 11173 .empty .empty) .empty) (.node 12589 .empty .empty)) (.node 13681 (.node 12979 .empty .empty) (.node 14437 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10657_ordered : SortedNatTree.ordered row10657 = true := by decide
set_option maxRecDepth 4096 in
theorem row10657_values : SortedNatTree.values row10657 = Data.row10657 := rfl
theorem row10657_checked : SortedNatTree.contains row10657 = Data.row10657.contains := by
  rw [SortedNatTree.contains_eq_list row10657 row10657_ordered, row10657_values]
def row10663 : SortedNatTree.Tree := (.node 13873 (.node 11161 (.node 10939 (.node 10909 .empty .empty) .empty) (.node 13291 (.node 13099 .empty .empty) .empty)) (.node 15031 (.node 14923 (.node 14461 .empty .empty) .empty) (.node 15199 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10663_ordered : SortedNatTree.ordered row10663 = true := by decide
set_option maxRecDepth 4096 in
theorem row10663_values : SortedNatTree.values row10663 = Data.row10663 := rfl
theorem row10663_checked : SortedNatTree.contains row10663 = Data.row10663.contains := by
  rw [SortedNatTree.contains_eq_list row10663 row10663_ordered, row10663_values]
def row10667 : SortedNatTree.Tree := (.node 13397 (.node 11351 (.node 10691 .empty .empty) (.node 12161 .empty .empty)) (.node 14771 (.node 14669 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row10667_ordered : SortedNatTree.ordered row10667 = true := by decide
set_option maxRecDepth 4096 in
theorem row10667_values : SortedNatTree.values row10667 = Data.row10667 := rfl
theorem row10667_checked : SortedNatTree.contains row10667 = Data.row10667.contains := by
  rw [SortedNatTree.contains_eq_list row10667 row10667_ordered, row10667_values]
def row10687 : SortedNatTree.Tree := (.node 13693 (.node 11779 (.node 11251 (.node 11047 .empty .empty) (.node 11593 .empty .empty)) (.node 12889 (.node 11971 .empty .empty) (.node 13309 .empty .empty))) (.node 14887 (.node 14653 (.node 14029 .empty .empty) (.node 14821 .empty .empty)) (.node 15139 (.node 14947 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row10687_ordered : SortedNatTree.ordered row10687 = true := by decide
set_option maxRecDepth 4096 in
theorem row10687_values : SortedNatTree.values row10687 = Data.row10687 := rfl
theorem row10687_checked : SortedNatTree.contains row10687 = Data.row10687.contains := by
  rw [SortedNatTree.contains_eq_list row10687 row10687_ordered, row10687_values]
def row10691 : SortedNatTree.Tree := (.node 14843 (.node 12227 (.node 11807 .empty .empty) (.node 12491 .empty .empty)) (.node 15263 (.node 15077 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row10691_ordered : SortedNatTree.ordered row10691 = true := by decide
set_option maxRecDepth 4096 in
theorem row10691_values : SortedNatTree.values row10691 = Data.row10691 := rfl
theorem row10691_checked : SortedNatTree.contains row10691 = Data.row10691.contains := by
  rw [SortedNatTree.contains_eq_list row10691 row10691_ordered, row10691_values]
def row10709 : SortedNatTree.Tree := (.node 12041 (.node 11489 (.node 11411 .empty .empty) (.node 11789 .empty .empty)) (.node 15173 (.node 14543 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row10709_ordered : SortedNatTree.ordered row10709 = true := by decide
set_option maxRecDepth 4096 in
theorem row10709_values : SortedNatTree.values row10709 = Data.row10709 := rfl
theorem row10709_checked : SortedNatTree.contains row10709 = Data.row10709.contains := by
  rw [SortedNatTree.contains_eq_list row10709 row10709_ordered, row10709_values]
def row10711 : SortedNatTree.Tree := (.node 12409 (.node 10753 (.node 10723 .empty .empty) (.node 11257 .empty .empty)) (.node 14293 (.node 13513 .empty .empty) (.node 14503 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10711_ordered : SortedNatTree.ordered row10711 = true := by decide
set_option maxRecDepth 4096 in
theorem row10711_values : SortedNatTree.values row10711 = Data.row10711 := rfl
theorem row10711_checked : SortedNatTree.contains row10711 = Data.row10711.contains := by
  rw [SortedNatTree.contains_eq_list row10711 row10711_ordered, row10711_values]
def row10723 : SortedNatTree.Tree := (.node 12697 (.node 12037 (.node 11887 (.node 10753 .empty .empty) .empty) (.node 12517 .empty .empty)) (.node 13873 (.node 13381 (.node 13003 .empty .empty) .empty) (.node 15289 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10723_ordered : SortedNatTree.ordered row10723 = true := by decide
set_option maxRecDepth 4096 in
theorem row10723_values : SortedNatTree.values row10723 = Data.row10723 := rfl
theorem row10723_checked : SortedNatTree.contains row10723 = Data.row10723.contains := by
  rw [SortedNatTree.contains_eq_list row10723 row10723_ordered, row10723_values]
def row10729 : SortedNatTree.Tree := (.node 11863 (.node 11617 (.node 10903 .empty .empty) .empty) (.node 12853 .empty .empty))
set_option maxRecDepth 4096 in
theorem row10729_ordered : SortedNatTree.ordered row10729 = true := by decide
set_option maxRecDepth 4096 in
theorem row10729_values : SortedNatTree.values row10729 = Data.row10729 := rfl
theorem row10729_checked : SortedNatTree.contains row10729 = Data.row10729.contains := by
  rw [SortedNatTree.contains_eq_list row10729 row10729_ordered, row10729_values]
def row10733 : SortedNatTree.Tree := (.node 11909 (.node 11783 (.node 10883 .empty .empty) .empty) (.node 14747 (.node 13313 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row10733_ordered : SortedNatTree.ordered row10733 = true := by decide
set_option maxRecDepth 4096 in
theorem row10733_values : SortedNatTree.values row10733 = Data.row10733 := rfl
theorem row10733_checked : SortedNatTree.contains row10733 = Data.row10733.contains := by
  rw [SortedNatTree.contains_eq_list row10733 row10733_ordered, row10733_values]
def row10739 : SortedNatTree.Tree := (.node 14423 (.node 12911 (.node 12479 .empty .empty) (.node 13331 .empty .empty)) (.node 14783 (.node 14717 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row10739_ordered : SortedNatTree.ordered row10739 = true := by decide
set_option maxRecDepth 4096 in
theorem row10739_values : SortedNatTree.values row10739 = Data.row10739 := rfl
theorem row10739_checked : SortedNatTree.contains row10739 = Data.row10739.contains := by
  rw [SortedNatTree.contains_eq_list row10739 row10739_ordered, row10739_values]
def row10753 : SortedNatTree.Tree := (.node 13033 (.node 11437 (.node 11119 (.node 10993 .empty .empty) .empty) (.node 12829 (.node 12409 .empty .empty) .empty)) (.node 14629 (.node 14437 (.node 13441 .empty .empty) .empty) (.node 14731 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10753_ordered : SortedNatTree.ordered row10753 = true := by decide
set_option maxRecDepth 4096 in
theorem row10753_values : SortedNatTree.values row10753 = Data.row10753 := rfl
theorem row10753_checked : SortedNatTree.contains row10753 = Data.row10753.contains := by
  rw [SortedNatTree.contains_eq_list row10753 row10753_ordered, row10753_values]
def row10771 : SortedNatTree.Tree := (.node 14011 (.node 13339 (.node 12409 .empty .empty) .empty) (.node 14767 .empty .empty))
set_option maxRecDepth 4096 in
theorem row10771_ordered : SortedNatTree.ordered row10771 = true := by decide
set_option maxRecDepth 4096 in
theorem row10771_values : SortedNatTree.values row10771 = Data.row10771 := rfl
theorem row10771_checked : SortedNatTree.contains row10771 = Data.row10771.contains := by
  rw [SortedNatTree.contains_eq_list row10771 row10771_ordered, row10771_values]
def row10781 : SortedNatTree.Tree := (.node 12491 (.node 11597 (.node 11369 (.node 10853 .empty .empty) .empty) (.node 12473 (.node 12437 .empty .empty) .empty)) (.node 13721 (.node 13499 (.node 12689 .empty .empty) .empty) (.node 14087 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10781_ordered : SortedNatTree.ordered row10781 = true := by decide
set_option maxRecDepth 4096 in
theorem row10781_values : SortedNatTree.values row10781 = Data.row10781 := rfl
theorem row10781_checked : SortedNatTree.contains row10781 = Data.row10781.contains := by
  rw [SortedNatTree.contains_eq_list row10781 row10781_ordered, row10781_values]
def row10789 : SortedNatTree.Tree := (.node 13963 (.node 12487 (.node 10939 .empty .empty) (.node 13009 .empty .empty)) (.node 14653 (.node 14461 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row10789_ordered : SortedNatTree.ordered row10789 = true := by decide
set_option maxRecDepth 4096 in
theorem row10789_values : SortedNatTree.values row10789 = Data.row10789 := rfl
theorem row10789_checked : SortedNatTree.contains row10789 = Data.row10789.contains := by
  rw [SortedNatTree.contains_eq_list row10789 row10789_ordered, row10789_values]
def row10799 : SortedNatTree.Tree := (.node 13577 (.node 12203 (.node 11681 .empty .empty) (.node 12269 .empty .empty)) (.node 13907 (.node 13829 .empty .empty) (.node 14057 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10799_ordered : SortedNatTree.ordered row10799 = true := by decide
set_option maxRecDepth 4096 in
theorem row10799_values : SortedNatTree.values row10799 = Data.row10799 := rfl
theorem row10799_checked : SortedNatTree.contains row10799 = Data.row10799.contains := by
  rw [SortedNatTree.contains_eq_list row10799 row10799_ordered, row10799_values]
def row10831 : SortedNatTree.Tree := (.node 14083 (.node 12073 (.node 12049 (.node 11119 .empty .empty) .empty) (.node 13711 .empty .empty)) (.node 14629 (.node 14341 .empty .empty) (.node 15091 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10831_ordered : SortedNatTree.ordered row10831 = true := by decide
set_option maxRecDepth 4096 in
theorem row10831_values : SortedNatTree.values row10831 = Data.row10831 := rfl
theorem row10831_checked : SortedNatTree.contains row10831 = Data.row10831.contains := by
  rw [SortedNatTree.contains_eq_list row10831 row10831_ordered, row10831_values]
def row10837 : SortedNatTree.Tree := (.node 13009 (.node 11887 (.node 10909 .empty .empty) (.node 12841 .empty .empty)) (.node 13723 (.node 13291 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row10837_ordered : SortedNatTree.ordered row10837 = true := by decide
set_option maxRecDepth 4096 in
theorem row10837_values : SortedNatTree.values row10837 = Data.row10837 := rfl
theorem row10837_checked : SortedNatTree.contains row10837 = Data.row10837.contains := by
  rw [SortedNatTree.contains_eq_list row10837 row10837_ordered, row10837_values]
def row10847 : SortedNatTree.Tree := (.node 13499 (.node 12263 (.node 11213 (.node 11057 .empty .empty) (.node 11681 .empty .empty)) (.node 13229 (.node 13037 .empty .empty) .empty)) (.node 14207 (.node 14153 (.node 13859 .empty .empty) .empty) (.node 15077 (.node 14879 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row10847_ordered : SortedNatTree.ordered row10847 = true := by decide
set_option maxRecDepth 4096 in
theorem row10847_values : SortedNatTree.values row10847 = Data.row10847 := rfl
theorem row10847_checked : SortedNatTree.contains row10847 = Data.row10847.contains := by
  rw [SortedNatTree.contains_eq_list row10847 row10847_ordered, row10847_values]
def row10853 : SortedNatTree.Tree := (.node 13367 (.node 11897 (.node 10937 .empty .empty) (.node 12251 .empty .empty)) (.node 15107 (.node 14207 .empty .empty) (.node 15137 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10853_ordered : SortedNatTree.ordered row10853 = true := by decide
set_option maxRecDepth 4096 in
theorem row10853_values : SortedNatTree.values row10853 = Data.row10853 := rfl
theorem row10853_checked : SortedNatTree.contains row10853 = Data.row10853.contains := by
  rw [SortedNatTree.contains_eq_list row10853 row10853_ordered, row10853_values]
def row10859 : SortedNatTree.Tree := (.node 13691 (.node 12143 (.node 11987 (.node 11471 .empty .empty) .empty) (.node 13463 (.node 12203 .empty .empty) .empty)) (.node 14957 (.node 14783 (.node 14033 .empty .empty) .empty) (.node 15161 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10859_ordered : SortedNatTree.ordered row10859 = true := by decide
set_option maxRecDepth 4096 in
theorem row10859_values : SortedNatTree.values row10859 = Data.row10859 := rfl
theorem row10859_checked : SortedNatTree.contains row10859 = Data.row10859.contains := by
  rw [SortedNatTree.contains_eq_list row10859 row10859_ordered, row10859_values]
def row10861 : SortedNatTree.Tree := (.node 12457 (.node 11287 (.node 10867 .empty .empty) (.node 11701 .empty .empty)) (.node 13417 (.node 12799 .empty .empty) (.node 14923 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10861_ordered : SortedNatTree.ordered row10861 = true := by decide
set_option maxRecDepth 4096 in
theorem row10861_values : SortedNatTree.values row10861 = Data.row10861 := rfl
theorem row10861_checked : SortedNatTree.contains row10861 = Data.row10861.contains := by
  rw [SortedNatTree.contains_eq_list row10861 row10861_ordered, row10861_values]
def row10867 : SortedNatTree.Tree := (.node 12589 .empty .empty)
set_option maxRecDepth 4096 in
theorem row10867_ordered : SortedNatTree.ordered row10867 = true := by decide
set_option maxRecDepth 4096 in
theorem row10867_values : SortedNatTree.values row10867 = Data.row10867 := rfl
theorem row10867_checked : SortedNatTree.contains row10867 = Data.row10867.contains := by
  rw [SortedNatTree.contains_eq_list row10867 row10867_ordered, row10867_values]
def row10883 : SortedNatTree.Tree := (.node 13619 (.node 13337 (.node 12227 (.node 11519 .empty .empty) .empty) (.node 13451 .empty .empty)) (.node 14831 (.node 13841 .empty .empty) (.node 14957 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10883_ordered : SortedNatTree.ordered row10883 = true := by decide
set_option maxRecDepth 4096 in
theorem row10883_values : SortedNatTree.values row10883 = Data.row10883 := rfl
theorem row10883_checked : SortedNatTree.contains row10883 = Data.row10883.contains := by
  rw [SortedNatTree.contains_eq_list row10883 row10883_ordered, row10883_values]
def row10889 : SortedNatTree.Tree := (.node 13229 (.node 12821 (.node 11903 (.node 11003 .empty .empty) (.node 11927 .empty .empty)) (.node 13109 (.node 12911 .empty .empty) .empty)) (.node 14081 (.node 13877 (.node 13577 .empty .empty) .empty) (.node 14543 (.node 14087 .empty .empty) .empty)))
set_option maxRecDepth 4096 in
theorem row10889_ordered : SortedNatTree.ordered row10889 = true := by decide
set_option maxRecDepth 4096 in
theorem row10889_values : SortedNatTree.values row10889 = Data.row10889 := rfl
theorem row10889_checked : SortedNatTree.contains row10889 = Data.row10889.contains := by
  rw [SortedNatTree.contains_eq_list row10889 row10889_ordered, row10889_values]
def row10891 : SortedNatTree.Tree := (.node 12511 (.node 11719 (.node 11701 (.node 11299 .empty .empty) .empty) (.node 12163 .empty .empty)) (.node 14419 (.node 12781 .empty .empty) (.node 14431 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10891_ordered : SortedNatTree.ordered row10891 = true := by decide
set_option maxRecDepth 4096 in
theorem row10891_values : SortedNatTree.values row10891 = Data.row10891 := rfl
theorem row10891_checked : SortedNatTree.contains row10891 = Data.row10891.contains := by
  rw [SortedNatTree.contains_eq_list row10891 row10891_ordered, row10891_values]
def row10903 : SortedNatTree.Tree := (.node 13381 (.node 12547 (.node 11353 .empty .empty) (.node 12583 .empty .empty)) (.node 14341 (.node 14083 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row10903_ordered : SortedNatTree.ordered row10903 = true := by decide
set_option maxRecDepth 4096 in
theorem row10903_values : SortedNatTree.values row10903 = Data.row10903 := rfl
theorem row10903_checked : SortedNatTree.contains row10903 = Data.row10903.contains := by
  rw [SortedNatTree.contains_eq_list row10903 row10903_ordered, row10903_values]
def row10909 : SortedNatTree.Tree := (.node 13687 (.node 12841 (.node 11257 .empty .empty) (.node 13597 .empty .empty)) (.node 14947 (.node 14437 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row10909_ordered : SortedNatTree.ordered row10909 = true := by decide
set_option maxRecDepth 4096 in
theorem row10909_values : SortedNatTree.values row10909 = Data.row10909 := rfl
theorem row10909_checked : SortedNatTree.contains row10909 = Data.row10909.contains := by
  rw [SortedNatTree.contains_eq_list row10909 row10909_ordered, row10909_values]
def row10937 : SortedNatTree.Tree := (.node 13679 (.node 11987 (.node 11447 (.node 11057 .empty .empty) .empty) (.node 12959 .empty .empty)) (.node 14639 (.node 14549 (.node 13997 .empty .empty) .empty) (.node 14969 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10937_ordered : SortedNatTree.ordered row10937 = true := by decide
set_option maxRecDepth 4096 in
theorem row10937_values : SortedNatTree.values row10937 = Data.row10937 := rfl
theorem row10937_checked : SortedNatTree.contains row10937 = Data.row10937.contains := by
  rw [SortedNatTree.contains_eq_list row10937 row10937_ordered, row10937_values]
def row10939 : SortedNatTree.Tree := (.node 12601 (.node 11827 (.node 11173 (.node 11131 .empty .empty) .empty) (.node 12577 .empty .empty)) (.node 13903 (.node 13591 (.node 13267 .empty .empty) .empty) (.node 14557 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10939_ordered : SortedNatTree.ordered row10939 = true := by decide
set_option maxRecDepth 4096 in
theorem row10939_values : SortedNatTree.values row10939 = Data.row10939 := rfl
theorem row10939_checked : SortedNatTree.contains row10939 = Data.row10939.contains := by
  rw [SortedNatTree.contains_eq_list row10939 row10939_ordered, row10939_values]
def row10949 : SortedNatTree.Tree := (.node 12641 (.node 11399 (.node 11369 .empty .empty) (.node 11903 .empty .empty)) (.node 13109 (.node 12809 .empty .empty) (.node 13499 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10949_ordered : SortedNatTree.ordered row10949 = true := by decide
set_option maxRecDepth 4096 in
theorem row10949_values : SortedNatTree.values row10949 = Data.row10949 := rfl
theorem row10949_checked : SortedNatTree.contains row10949 = Data.row10949.contains := by
  rw [SortedNatTree.contains_eq_list row10949 row10949_ordered, row10949_values]
def row10957 : SortedNatTree.Tree := (.node 14341 (.node 13183 (.node 12613 .empty .empty) .empty) (.node 14563 .empty .empty))
set_option maxRecDepth 4096 in
theorem row10957_ordered : SortedNatTree.ordered row10957 = true := by decide
set_option maxRecDepth 4096 in
theorem row10957_values : SortedNatTree.values row10957 = Data.row10957 := rfl
theorem row10957_checked : SortedNatTree.contains row10957 = Data.row10957.contains := by
  rw [SortedNatTree.contains_eq_list row10957 row10957_ordered, row10957_values]
def row10973 : SortedNatTree.Tree := (.node 13577 (.node 12569 (.node 12143 (.node 11939 .empty .empty) .empty) (.node 12659 .empty .empty)) (.node 13901 (.node 13781 .empty .empty) (.node 14879 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10973_ordered : SortedNatTree.ordered row10973 = true := by decide
set_option maxRecDepth 4096 in
theorem row10973_values : SortedNatTree.values row10973 = Data.row10973 := rfl
theorem row10973_checked : SortedNatTree.contains row10973 = Data.row10973.contains := by
  rw [SortedNatTree.contains_eq_list row10973 row10973_ordered, row10973_values]
def row10979 : SortedNatTree.Tree := (.node 13259 (.node 12401 (.node 11987 (.node 11621 .empty .empty) .empty) (.node 12671 .empty .empty)) (.node 14843 (.node 14771 .empty .empty) (.node 15053 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10979_ordered : SortedNatTree.ordered row10979 = true := by decide
set_option maxRecDepth 4096 in
theorem row10979_values : SortedNatTree.values row10979 = Data.row10979 := rfl
theorem row10979_checked : SortedNatTree.contains row10979 = Data.row10979.contains := by
  rw [SortedNatTree.contains_eq_list row10979 row10979_ordered, row10979_values]
def row10987 : SortedNatTree.Tree := (.node 12409 (.node 11743 (.node 11701 (.node 11491 .empty .empty) .empty) (.node 12379 .empty .empty)) (.node 13003 (.node 12613 .empty .empty) (.node 13999 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10987_ordered : SortedNatTree.ordered row10987 = true := by decide
set_option maxRecDepth 4096 in
theorem row10987_values : SortedNatTree.values row10987 = Data.row10987 := rfl
theorem row10987_checked : SortedNatTree.contains row10987 = Data.row10987.contains := by
  rw [SortedNatTree.contains_eq_list row10987 row10987_ordered, row10987_values]
def row10993 : SortedNatTree.Tree := (.node 13183 (.node 12589 (.node 11593 (.node 11173 .empty .empty) .empty) (.node 13009 .empty .empty)) (.node 14851 (.node 14347 (.node 13807 .empty .empty) .empty) (.node 14947 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row10993_ordered : SortedNatTree.ordered row10993 = true := by decide
set_option maxRecDepth 4096 in
theorem row10993_values : SortedNatTree.values row10993 = Data.row10993 := rfl
theorem row10993_checked : SortedNatTree.contains row10993 = Data.row10993.contains := by
  rw [SortedNatTree.contains_eq_list row10993 row10993_ordered, row10993_values]
def row11003 : SortedNatTree.Tree := (.node 12119 (.node 11579 (.node 11321 .empty .empty) (.node 11657 .empty .empty)) (.node 13523 (.node 13127 .empty .empty) (.node 14249 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row11003_ordered : SortedNatTree.ordered row11003 = true := by decide
set_option maxRecDepth 4096 in
theorem row11003_values : SortedNatTree.values row11003 = Data.row11003 := rfl
theorem row11003_checked : SortedNatTree.contains row11003 = Data.row11003.contains := by
  rw [SortedNatTree.contains_eq_list row11003 row11003_ordered, row11003_values]
def row11027 : SortedNatTree.Tree := (.node 13037 (.node 12491 (.node 11903 (.node 11087 .empty .empty) .empty) (.node 12911 .empty .empty)) (.node 14321 (.node 13229 (.node 13151 .empty .empty) .empty) (.node 14867 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row11027_ordered : SortedNatTree.ordered row11027 = true := by decide
set_option maxRecDepth 4096 in
theorem row11027_values : SortedNatTree.values row11027 = Data.row11027 := rfl
theorem row11027_checked : SortedNatTree.contains row11027 = Data.row11027.contains := by
  rw [SortedNatTree.contains_eq_list row11027 row11027_ordered, row11027_values]
def row11047 : SortedNatTree.Tree := (.node 13933 (.node 12277 (.node 11677 .empty .empty) .empty) (.node 14293 .empty .empty))
set_option maxRecDepth 4096 in
theorem row11047_ordered : SortedNatTree.ordered row11047 = true := by decide
set_option maxRecDepth 4096 in
theorem row11047_values : SortedNatTree.values row11047 = Data.row11047 := rfl
theorem row11047_checked : SortedNatTree.contains row11047 = Data.row11047.contains := by
  rw [SortedNatTree.contains_eq_list row11047 row11047_ordered, row11047_values]
def row11057 : SortedNatTree.Tree := (.node 13241 (.node 11969 (.node 11939 (.node 11351 .empty .empty) .empty) (.node 12197 (.node 12071 .empty .empty) .empty)) (.node 14741 (.node 14669 (.node 14369 .empty .empty) .empty) (.node 14759 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row11057_ordered : SortedNatTree.ordered row11057 = true := by decide
set_option maxRecDepth 4096 in
theorem row11057_values : SortedNatTree.values row11057 = Data.row11057 := rfl
theorem row11057_checked : SortedNatTree.contains row11057 = Data.row11057.contains := by
  rw [SortedNatTree.contains_eq_list row11057 row11057_ordered, row11057_values]
def row11059 : SortedNatTree.Tree := (.node 12163 (.node 11437 (.node 11239 .empty .empty) (.node 11731 .empty .empty)) (.node 13789 (.node 12739 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row11059_ordered : SortedNatTree.ordered row11059 = true := by decide
set_option maxRecDepth 4096 in
theorem row11059_values : SortedNatTree.values row11059 = Data.row11059 := rfl
theorem row11059_checked : SortedNatTree.contains row11059 = Data.row11059.contains := by
  rw [SortedNatTree.contains_eq_list row11059 row11059_ordered, row11059_values]
def row11069 : SortedNatTree.Tree := (.node 12689 (.node 11279 (.node 11171 .empty .empty) (.node 12269 .empty .empty)) (.node 14153 (.node 12941 .empty .empty) (.node 14423 .empty .empty)))
set_option maxRecDepth 4096 in
theorem row11069_ordered : SortedNatTree.ordered row11069 = true := by decide
set_option maxRecDepth 4096 in
theorem row11069_values : SortedNatTree.values row11069 = Data.row11069 := rfl
theorem row11069_checked : SortedNatTree.contains row11069 = Data.row11069.contains := by
  rw [SortedNatTree.contains_eq_list row11069 row11069_ordered, row11069_values]
def row11071 : SortedNatTree.Tree := (.node 12073 (.node 11491 (.node 11443 .empty .empty) .empty) (.node 14293 (.node 13291 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row11071_ordered : SortedNatTree.ordered row11071 = true := by decide
set_option maxRecDepth 4096 in
theorem row11071_values : SortedNatTree.values row11071 = Data.row11071 := rfl
theorem row11071_checked : SortedNatTree.contains row11071 = Data.row11071.contains := by
  rw [SortedNatTree.contains_eq_list row11071 row11071_ordered, row11071_values]
def row11083 : SortedNatTree.Tree := (.node 12211 (.node 11587 (.node 11353 .empty .empty) .empty) (.node 14923 .empty .empty))
set_option maxRecDepth 4096 in
theorem row11083_ordered : SortedNatTree.ordered row11083 = true := by decide
set_option maxRecDepth 4096 in
theorem row11083_values : SortedNatTree.values row11083 = Data.row11083 := rfl
theorem row11083_checked : SortedNatTree.contains row11083 = Data.row11083.contains := by
  rw [SortedNatTree.contains_eq_list row11083 row11083_ordered, row11083_values]
def row11087 : SortedNatTree.Tree := (.node 12959 (.node 11939 (.node 11867 .empty .empty) .empty) (.node 14561 (.node 13049 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row11087_ordered : SortedNatTree.ordered row11087 = true := by decide
set_option maxRecDepth 4096 in
theorem row11087_values : SortedNatTree.values row11087 = Data.row11087 := rfl
theorem row11087_checked : SortedNatTree.contains row11087 = Data.row11087.contains := by
  rw [SortedNatTree.contains_eq_list row11087 row11087_ordered, row11087_values]
def row11093 : SortedNatTree.Tree := (.node 13367 (.node 12143 (.node 11981 .empty .empty) (.node 13151 .empty .empty)) (.node 14813 (.node 14303 .empty .empty) .empty))
set_option maxRecDepth 4096 in
theorem row11093_ordered : SortedNatTree.ordered row11093 = true := by decide
set_option maxRecDepth 4096 in
theorem row11093_values : SortedNatTree.values row11093 = Data.row11093 := rfl
theorem row11093_checked : SortedNatTree.contains row11093 = Data.row11093.contains := by
  rw [SortedNatTree.contains_eq_list row11093 row11093_ordered, row11093_values]
def row11113 : SortedNatTree.Tree := (.node 13873 (.node 12841 (.node 12739 .empty .empty) .empty) (.node 14419 .empty .empty))
set_option maxRecDepth 4096 in
theorem row11113_ordered : SortedNatTree.ordered row11113 = true := by decide
set_option maxRecDepth 4096 in
theorem row11113_values : SortedNatTree.values row11113 = Data.row11113 := rfl
theorem row11113_checked : SortedNatTree.contains row11113 = Data.row11113.contains := by
  rw [SortedNatTree.contains_eq_list row11113 row11113_ordered, row11113_values]
end ProjectEulerSolutions.PrimePairLarge.FastRows
