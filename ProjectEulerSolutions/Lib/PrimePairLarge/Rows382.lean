import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row12841_small : List Nat := []
theorem row12841_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12841 row12841_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12841_block_005 : List Nat := []
theorem row12841_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12841 row12841_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12841_block_006 : List Nat := []
theorem row12841_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12841 row12841_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12841_block_007 : List Nat := []
def plan12841_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 11), .second, .first, .first, (.firstFactor 17), .first, (.firstFactor 37), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 31), .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12841_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12841 row12841_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12841_block_007
  · decide
  · rfl

def row12841_block_008 : List Nat := []
theorem row12841_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12841 row12841_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12841_block_009 : List Nat := []
theorem row12841_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12841 row12841_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12841_block_010 : List Nat := []
theorem row12841_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12841 row12841_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12841_block_011 : List Nat := []
theorem row12841_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12841 row12841_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12841_block_012 : List Nat := []
theorem row12841_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12841 row12841_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12841_block_013 : List Nat := []
theorem row12841_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12841 row12841_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12841_large_left_left : List Nat := row12841_small ++ row12841_block_005
theorem row12841_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12841 row12841_large_left_left :=
  Budget.RowUpperBound.branch row12841_small_covers row12841_block_005_covers

def row12841_large_left_right_right : List Nat := row12841_block_007 ++ row12841_block_008
theorem row12841_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12841 row12841_large_left_right_right :=
  Budget.RowUpperBound.branch row12841_block_007_covers row12841_block_008_covers

def row12841_large_left_right : List Nat := row12841_block_006 ++ row12841_large_left_right_right
theorem row12841_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12841 row12841_large_left_right :=
  Budget.RowUpperBound.branch row12841_block_006_covers row12841_large_left_right_right_covers

def row12841_large_left : List Nat := row12841_large_left_left ++ row12841_large_left_right
theorem row12841_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12841 row12841_large_left :=
  Budget.RowUpperBound.branch row12841_large_left_left_covers row12841_large_left_right_covers

def row12841_large_right_left : List Nat := row12841_block_009 ++ row12841_block_010
theorem row12841_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12841 row12841_large_right_left :=
  Budget.RowUpperBound.branch row12841_block_009_covers row12841_block_010_covers

def row12841_large_right_right_right : List Nat := row12841_block_012 ++ row12841_block_013
theorem row12841_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12841 row12841_large_right_right_right :=
  Budget.RowUpperBound.branch row12841_block_012_covers row12841_block_013_covers

def row12841_large_right_right : List Nat := row12841_block_011 ++ row12841_large_right_right_right
theorem row12841_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12841 row12841_large_right_right :=
  Budget.RowUpperBound.branch row12841_block_011_covers row12841_large_right_right_right_covers

def row12841_large_right : List Nat := row12841_large_right_left ++ row12841_large_right_right
theorem row12841_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12841 row12841_large_right :=
  Budget.RowUpperBound.branch row12841_large_right_left_covers row12841_large_right_right_covers

def row12841_large : List Nat := row12841_large_left ++ row12841_large_right
theorem row12841_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12841 row12841_large :=
  Budget.RowUpperBound.branch row12841_large_left_covers row12841_large_right_covers

def row12841 : List Nat := row12841_large
theorem row12841_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12841 row12841 := row12841_large_covers

def row12853_small : List Nat := []
theorem row12853_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12853 row12853_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12853_block_005 : List Nat := []
theorem row12853_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12853 row12853_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12853_block_006 : List Nat := []
theorem row12853_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12853 row12853_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12853_block_007 : List Nat := []
def plan12853_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12853_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12853 row12853_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12853_block_007
  · decide
  · rfl

def row12853_block_008 : List Nat := []
theorem row12853_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12853 row12853_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12853_block_009 : List Nat := []
theorem row12853_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12853 row12853_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12853_block_010 : List Nat := []
theorem row12853_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12853 row12853_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12853_block_011 : List Nat := []
theorem row12853_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12853 row12853_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12853_block_012 : List Nat := []
theorem row12853_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12853 row12853_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12853_block_013 : List Nat := []
theorem row12853_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12853 row12853_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12853_large_left_left : List Nat := row12853_small ++ row12853_block_005
theorem row12853_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12853 row12853_large_left_left :=
  Budget.RowUpperBound.branch row12853_small_covers row12853_block_005_covers

def row12853_large_left_right_right : List Nat := row12853_block_007 ++ row12853_block_008
theorem row12853_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12853 row12853_large_left_right_right :=
  Budget.RowUpperBound.branch row12853_block_007_covers row12853_block_008_covers

def row12853_large_left_right : List Nat := row12853_block_006 ++ row12853_large_left_right_right
theorem row12853_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12853 row12853_large_left_right :=
  Budget.RowUpperBound.branch row12853_block_006_covers row12853_large_left_right_right_covers

def row12853_large_left : List Nat := row12853_large_left_left ++ row12853_large_left_right
theorem row12853_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12853 row12853_large_left :=
  Budget.RowUpperBound.branch row12853_large_left_left_covers row12853_large_left_right_covers

def row12853_large_right_left : List Nat := row12853_block_009 ++ row12853_block_010
theorem row12853_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12853 row12853_large_right_left :=
  Budget.RowUpperBound.branch row12853_block_009_covers row12853_block_010_covers

def row12853_large_right_right_right : List Nat := row12853_block_012 ++ row12853_block_013
theorem row12853_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12853 row12853_large_right_right_right :=
  Budget.RowUpperBound.branch row12853_block_012_covers row12853_block_013_covers

def row12853_large_right_right : List Nat := row12853_block_011 ++ row12853_large_right_right_right
theorem row12853_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12853 row12853_large_right_right :=
  Budget.RowUpperBound.branch row12853_block_011_covers row12853_large_right_right_right_covers

def row12853_large_right : List Nat := row12853_large_right_left ++ row12853_large_right_right
theorem row12853_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12853 row12853_large_right :=
  Budget.RowUpperBound.branch row12853_large_right_left_covers row12853_large_right_right_covers

def row12853_large : List Nat := row12853_large_left ++ row12853_large_right
theorem row12853_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12853 row12853_large :=
  Budget.RowUpperBound.branch row12853_large_left_covers row12853_large_right_covers

def row12853 : List Nat := row12853_large
theorem row12853_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12853 row12853 := row12853_large_covers

def row12889_small : List Nat := []
theorem row12889_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12889 row12889_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12889_block_005 : List Nat := []
theorem row12889_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12889 row12889_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12889_block_006 : List Nat := []
theorem row12889_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12889 row12889_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12889_block_007 : List Nat := []
def plan12889_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.secondFactor 29), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .second, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 31), (.firstFactor 29), .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .second, .first, .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12889_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12889 row12889_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12889_block_007
  · decide
  · rfl

def row12889_block_008 : List Nat := []
theorem row12889_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12889 row12889_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12889_block_009 : List Nat := []
theorem row12889_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12889 row12889_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12889_block_010 : List Nat := []
theorem row12889_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12889 row12889_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12889_block_011 : List Nat := []
theorem row12889_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12889 row12889_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12889_block_012 : List Nat := []
theorem row12889_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12889 row12889_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12889_block_013 : List Nat := []
theorem row12889_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12889 row12889_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12889_large_left_left : List Nat := row12889_small ++ row12889_block_005
theorem row12889_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12889 row12889_large_left_left :=
  Budget.RowUpperBound.branch row12889_small_covers row12889_block_005_covers

def row12889_large_left_right_right : List Nat := row12889_block_007 ++ row12889_block_008
theorem row12889_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12889 row12889_large_left_right_right :=
  Budget.RowUpperBound.branch row12889_block_007_covers row12889_block_008_covers

def row12889_large_left_right : List Nat := row12889_block_006 ++ row12889_large_left_right_right
theorem row12889_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12889 row12889_large_left_right :=
  Budget.RowUpperBound.branch row12889_block_006_covers row12889_large_left_right_right_covers

def row12889_large_left : List Nat := row12889_large_left_left ++ row12889_large_left_right
theorem row12889_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12889 row12889_large_left :=
  Budget.RowUpperBound.branch row12889_large_left_left_covers row12889_large_left_right_covers

def row12889_large_right_left : List Nat := row12889_block_009 ++ row12889_block_010
theorem row12889_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12889 row12889_large_right_left :=
  Budget.RowUpperBound.branch row12889_block_009_covers row12889_block_010_covers

def row12889_large_right_right_right : List Nat := row12889_block_012 ++ row12889_block_013
theorem row12889_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12889 row12889_large_right_right_right :=
  Budget.RowUpperBound.branch row12889_block_012_covers row12889_block_013_covers

def row12889_large_right_right : List Nat := row12889_block_011 ++ row12889_large_right_right_right
theorem row12889_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12889 row12889_large_right_right :=
  Budget.RowUpperBound.branch row12889_block_011_covers row12889_large_right_right_right_covers

def row12889_large_right : List Nat := row12889_large_right_left ++ row12889_large_right_right
theorem row12889_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12889 row12889_large_right :=
  Budget.RowUpperBound.branch row12889_large_right_left_covers row12889_large_right_right_covers

def row12889_large : List Nat := row12889_large_left ++ row12889_large_right
theorem row12889_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12889 row12889_large :=
  Budget.RowUpperBound.branch row12889_large_left_covers row12889_large_right_covers

def row12889 : List Nat := row12889_large
theorem row12889_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12889 row12889 := row12889_large_covers

def row12893_small : List Nat := []
theorem row12893_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12893 row12893_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12893_block_005 : List Nat := []
theorem row12893_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12893 row12893_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12893_block_006 : List Nat := []
theorem row12893_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12893 row12893_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12893_block_007 : List Nat := [13049]
def plan12893_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 7), .first, (.secondFactor 7), .second, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 13), .possible, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 7), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12893_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12893 row12893_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12893_block_007
  · decide
  · rfl

def row12893_block_008 : List Nat := []
theorem row12893_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12893 row12893_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12893_block_009 : List Nat := []
theorem row12893_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12893 row12893_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12893_block_010 : List Nat := []
theorem row12893_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12893 row12893_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12893_block_011 : List Nat := []
theorem row12893_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12893 row12893_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12893_block_012 : List Nat := []
theorem row12893_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12893 row12893_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12893_block_013 : List Nat := []
theorem row12893_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12893 row12893_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12893_large_left_left : List Nat := row12893_small ++ row12893_block_005
theorem row12893_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12893 row12893_large_left_left :=
  Budget.RowUpperBound.branch row12893_small_covers row12893_block_005_covers

def row12893_large_left_right_right : List Nat := row12893_block_007 ++ row12893_block_008
theorem row12893_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12893 row12893_large_left_right_right :=
  Budget.RowUpperBound.branch row12893_block_007_covers row12893_block_008_covers

def row12893_large_left_right : List Nat := row12893_block_006 ++ row12893_large_left_right_right
theorem row12893_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12893 row12893_large_left_right :=
  Budget.RowUpperBound.branch row12893_block_006_covers row12893_large_left_right_right_covers

def row12893_large_left : List Nat := row12893_large_left_left ++ row12893_large_left_right
theorem row12893_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12893 row12893_large_left :=
  Budget.RowUpperBound.branch row12893_large_left_left_covers row12893_large_left_right_covers

def row12893_large_right_left : List Nat := row12893_block_009 ++ row12893_block_010
theorem row12893_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12893 row12893_large_right_left :=
  Budget.RowUpperBound.branch row12893_block_009_covers row12893_block_010_covers

def row12893_large_right_right_right : List Nat := row12893_block_012 ++ row12893_block_013
theorem row12893_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12893 row12893_large_right_right_right :=
  Budget.RowUpperBound.branch row12893_block_012_covers row12893_block_013_covers

def row12893_large_right_right : List Nat := row12893_block_011 ++ row12893_large_right_right_right
theorem row12893_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12893 row12893_large_right_right :=
  Budget.RowUpperBound.branch row12893_block_011_covers row12893_large_right_right_right_covers

def row12893_large_right : List Nat := row12893_large_right_left ++ row12893_large_right_right
theorem row12893_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12893 row12893_large_right :=
  Budget.RowUpperBound.branch row12893_large_right_left_covers row12893_large_right_right_covers

def row12893_large : List Nat := row12893_large_left ++ row12893_large_right
theorem row12893_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12893 row12893_large :=
  Budget.RowUpperBound.branch row12893_large_left_covers row12893_large_right_covers

def row12893 : List Nat := row12893_large
theorem row12893_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12893 row12893 := row12893_large_covers

end ProjectEulerSolutions.PrimePairLarge
