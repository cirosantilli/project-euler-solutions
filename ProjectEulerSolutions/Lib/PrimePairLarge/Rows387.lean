import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row13009_small : List Nat := []
theorem row13009_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 13009 row13009_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row13009_block_005 : List Nat := []
theorem row13009_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 13009 row13009_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row13009_block_006 : List Nat := []
theorem row13009_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 13009 row13009_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row13009_block_007 : List Nat := []
def plan13009_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row13009_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 13009 row13009_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan13009_block_007
  · decide
  · rfl

def row13009_block_008 : List Nat := []
theorem row13009_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 13009 row13009_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row13009_block_009 : List Nat := []
theorem row13009_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 13009 row13009_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row13009_block_010 : List Nat := []
theorem row13009_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 13009 row13009_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row13009_block_011 : List Nat := []
theorem row13009_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 13009 row13009_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row13009_block_012 : List Nat := []
theorem row13009_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 13009 row13009_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row13009_block_013 : List Nat := []
theorem row13009_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 13009 row13009_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row13009_large_left_left : List Nat := row13009_small ++ row13009_block_005
theorem row13009_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 13009 row13009_large_left_left :=
  Budget.RowUpperBound.branch row13009_small_covers row13009_block_005_covers

def row13009_large_left_right_right : List Nat := row13009_block_007 ++ row13009_block_008
theorem row13009_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 13009 row13009_large_left_right_right :=
  Budget.RowUpperBound.branch row13009_block_007_covers row13009_block_008_covers

def row13009_large_left_right : List Nat := row13009_block_006 ++ row13009_large_left_right_right
theorem row13009_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 13009 row13009_large_left_right :=
  Budget.RowUpperBound.branch row13009_block_006_covers row13009_large_left_right_right_covers

def row13009_large_left : List Nat := row13009_large_left_left ++ row13009_large_left_right
theorem row13009_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 13009 row13009_large_left :=
  Budget.RowUpperBound.branch row13009_large_left_left_covers row13009_large_left_right_covers

def row13009_large_right_left : List Nat := row13009_block_009 ++ row13009_block_010
theorem row13009_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 13009 row13009_large_right_left :=
  Budget.RowUpperBound.branch row13009_block_009_covers row13009_block_010_covers

def row13009_large_right_right_right : List Nat := row13009_block_012 ++ row13009_block_013
theorem row13009_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 13009 row13009_large_right_right_right :=
  Budget.RowUpperBound.branch row13009_block_012_covers row13009_block_013_covers

def row13009_large_right_right : List Nat := row13009_block_011 ++ row13009_large_right_right_right
theorem row13009_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 13009 row13009_large_right_right :=
  Budget.RowUpperBound.branch row13009_block_011_covers row13009_large_right_right_right_covers

def row13009_large_right : List Nat := row13009_large_right_left ++ row13009_large_right_right
theorem row13009_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 13009 row13009_large_right :=
  Budget.RowUpperBound.branch row13009_large_right_left_covers row13009_large_right_right_covers

def row13009_large : List Nat := row13009_large_left ++ row13009_large_right
theorem row13009_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 13009 row13009_large :=
  Budget.RowUpperBound.branch row13009_large_left_covers row13009_large_right_covers

def row13009 : List Nat := row13009_large
theorem row13009_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 13009 row13009 := row13009_large_covers

end ProjectEulerSolutions.PrimePairLarge
