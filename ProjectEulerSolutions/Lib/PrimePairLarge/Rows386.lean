import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row12983_small : List Nat := []
theorem row12983_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12983 row12983_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12983_block_005 : List Nat := []
theorem row12983_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12983 row12983_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12983_block_006 : List Nat := []
theorem row12983_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12983 row12983_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12983_block_007 : List Nat := []
def plan12983_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12983_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12983 row12983_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12983_block_007
  · decide
  · rfl

def row12983_block_008 : List Nat := []
theorem row12983_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12983 row12983_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12983_block_009 : List Nat := []
theorem row12983_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12983 row12983_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12983_block_010 : List Nat := []
theorem row12983_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12983 row12983_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12983_block_011 : List Nat := []
theorem row12983_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12983 row12983_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12983_block_012 : List Nat := []
theorem row12983_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12983 row12983_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12983_block_013 : List Nat := []
theorem row12983_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12983 row12983_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12983_large_left_left : List Nat := row12983_small ++ row12983_block_005
theorem row12983_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12983 row12983_large_left_left :=
  Budget.RowUpperBound.branch row12983_small_covers row12983_block_005_covers

def row12983_large_left_right_right : List Nat := row12983_block_007 ++ row12983_block_008
theorem row12983_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12983 row12983_large_left_right_right :=
  Budget.RowUpperBound.branch row12983_block_007_covers row12983_block_008_covers

def row12983_large_left_right : List Nat := row12983_block_006 ++ row12983_large_left_right_right
theorem row12983_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12983 row12983_large_left_right :=
  Budget.RowUpperBound.branch row12983_block_006_covers row12983_large_left_right_right_covers

def row12983_large_left : List Nat := row12983_large_left_left ++ row12983_large_left_right
theorem row12983_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12983 row12983_large_left :=
  Budget.RowUpperBound.branch row12983_large_left_left_covers row12983_large_left_right_covers

def row12983_large_right_left : List Nat := row12983_block_009 ++ row12983_block_010
theorem row12983_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12983 row12983_large_right_left :=
  Budget.RowUpperBound.branch row12983_block_009_covers row12983_block_010_covers

def row12983_large_right_right_right : List Nat := row12983_block_012 ++ row12983_block_013
theorem row12983_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12983 row12983_large_right_right_right :=
  Budget.RowUpperBound.branch row12983_block_012_covers row12983_block_013_covers

def row12983_large_right_right : List Nat := row12983_block_011 ++ row12983_large_right_right_right
theorem row12983_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12983 row12983_large_right_right :=
  Budget.RowUpperBound.branch row12983_block_011_covers row12983_large_right_right_right_covers

def row12983_large_right : List Nat := row12983_large_right_left ++ row12983_large_right_right
theorem row12983_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12983 row12983_large_right :=
  Budget.RowUpperBound.branch row12983_large_right_left_covers row12983_large_right_right_covers

def row12983_large : List Nat := row12983_large_left ++ row12983_large_right
theorem row12983_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12983 row12983_large :=
  Budget.RowUpperBound.branch row12983_large_left_covers row12983_large_right_covers

def row12983 : List Nat := row12983_large
theorem row12983_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12983 row12983 := row12983_large_covers

def row13001_small : List Nat := []
theorem row13001_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 13001 row13001_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row13001_block_005 : List Nat := []
theorem row13001_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 13001 row13001_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row13001_block_006 : List Nat := []
theorem row13001_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 13001 row13001_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row13001_block_007 : List Nat := []
def plan13001_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 29), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row13001_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 13001 row13001_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan13001_block_007
  · decide
  · rfl

def row13001_block_008 : List Nat := []
theorem row13001_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 13001 row13001_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row13001_block_009 : List Nat := []
theorem row13001_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 13001 row13001_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row13001_block_010 : List Nat := []
theorem row13001_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 13001 row13001_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row13001_block_011 : List Nat := []
theorem row13001_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 13001 row13001_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row13001_block_012 : List Nat := []
theorem row13001_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 13001 row13001_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row13001_block_013 : List Nat := []
theorem row13001_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 13001 row13001_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row13001_large_left_left : List Nat := row13001_small ++ row13001_block_005
theorem row13001_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 13001 row13001_large_left_left :=
  Budget.RowUpperBound.branch row13001_small_covers row13001_block_005_covers

def row13001_large_left_right_right : List Nat := row13001_block_007 ++ row13001_block_008
theorem row13001_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 13001 row13001_large_left_right_right :=
  Budget.RowUpperBound.branch row13001_block_007_covers row13001_block_008_covers

def row13001_large_left_right : List Nat := row13001_block_006 ++ row13001_large_left_right_right
theorem row13001_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 13001 row13001_large_left_right :=
  Budget.RowUpperBound.branch row13001_block_006_covers row13001_large_left_right_right_covers

def row13001_large_left : List Nat := row13001_large_left_left ++ row13001_large_left_right
theorem row13001_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 13001 row13001_large_left :=
  Budget.RowUpperBound.branch row13001_large_left_left_covers row13001_large_left_right_covers

def row13001_large_right_left : List Nat := row13001_block_009 ++ row13001_block_010
theorem row13001_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 13001 row13001_large_right_left :=
  Budget.RowUpperBound.branch row13001_block_009_covers row13001_block_010_covers

def row13001_large_right_right_right : List Nat := row13001_block_012 ++ row13001_block_013
theorem row13001_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 13001 row13001_large_right_right_right :=
  Budget.RowUpperBound.branch row13001_block_012_covers row13001_block_013_covers

def row13001_large_right_right : List Nat := row13001_block_011 ++ row13001_large_right_right_right
theorem row13001_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 13001 row13001_large_right_right :=
  Budget.RowUpperBound.branch row13001_block_011_covers row13001_large_right_right_right_covers

def row13001_large_right : List Nat := row13001_large_right_left ++ row13001_large_right_right
theorem row13001_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 13001 row13001_large_right :=
  Budget.RowUpperBound.branch row13001_large_right_left_covers row13001_large_right_right_covers

def row13001_large : List Nat := row13001_large_left ++ row13001_large_right
theorem row13001_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 13001 row13001_large :=
  Budget.RowUpperBound.branch row13001_large_left_covers row13001_large_right_covers

def row13001 : List Nat := row13001_large
theorem row13001_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 13001 row13001 := row13001_large_covers

def row13003_small : List Nat := []
theorem row13003_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 13003 row13003_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row13003_block_005 : List Nat := []
theorem row13003_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 13003 row13003_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row13003_block_006 : List Nat := []
theorem row13003_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 13003 row13003_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row13003_block_007 : List Nat := []
def plan13003_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.secondFactor 13), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row13003_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 13003 row13003_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan13003_block_007
  · decide
  · rfl

def row13003_block_008 : List Nat := []
theorem row13003_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 13003 row13003_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row13003_block_009 : List Nat := []
theorem row13003_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 13003 row13003_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row13003_block_010 : List Nat := []
theorem row13003_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 13003 row13003_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row13003_block_011 : List Nat := []
theorem row13003_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 13003 row13003_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row13003_block_012 : List Nat := []
theorem row13003_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 13003 row13003_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row13003_block_013 : List Nat := []
theorem row13003_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 13003 row13003_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row13003_large_left_left : List Nat := row13003_small ++ row13003_block_005
theorem row13003_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 13003 row13003_large_left_left :=
  Budget.RowUpperBound.branch row13003_small_covers row13003_block_005_covers

def row13003_large_left_right_right : List Nat := row13003_block_007 ++ row13003_block_008
theorem row13003_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 13003 row13003_large_left_right_right :=
  Budget.RowUpperBound.branch row13003_block_007_covers row13003_block_008_covers

def row13003_large_left_right : List Nat := row13003_block_006 ++ row13003_large_left_right_right
theorem row13003_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 13003 row13003_large_left_right :=
  Budget.RowUpperBound.branch row13003_block_006_covers row13003_large_left_right_right_covers

def row13003_large_left : List Nat := row13003_large_left_left ++ row13003_large_left_right
theorem row13003_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 13003 row13003_large_left :=
  Budget.RowUpperBound.branch row13003_large_left_left_covers row13003_large_left_right_covers

def row13003_large_right_left : List Nat := row13003_block_009 ++ row13003_block_010
theorem row13003_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 13003 row13003_large_right_left :=
  Budget.RowUpperBound.branch row13003_block_009_covers row13003_block_010_covers

def row13003_large_right_right_right : List Nat := row13003_block_012 ++ row13003_block_013
theorem row13003_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 13003 row13003_large_right_right_right :=
  Budget.RowUpperBound.branch row13003_block_012_covers row13003_block_013_covers

def row13003_large_right_right : List Nat := row13003_block_011 ++ row13003_large_right_right_right
theorem row13003_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 13003 row13003_large_right_right :=
  Budget.RowUpperBound.branch row13003_block_011_covers row13003_large_right_right_right_covers

def row13003_large_right : List Nat := row13003_large_right_left ++ row13003_large_right_right
theorem row13003_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 13003 row13003_large_right :=
  Budget.RowUpperBound.branch row13003_large_right_left_covers row13003_large_right_right_covers

def row13003_large : List Nat := row13003_large_left ++ row13003_large_right
theorem row13003_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 13003 row13003_large :=
  Budget.RowUpperBound.branch row13003_large_left_covers row13003_large_right_covers

def row13003 : List Nat := row13003_large
theorem row13003_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 13003 row13003 := row13003_large_covers

def row13007_small : List Nat := []
theorem row13007_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 13007 row13007_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row13007_block_005 : List Nat := []
theorem row13007_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 13007 row13007_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row13007_block_006 : List Nat := []
theorem row13007_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 13007 row13007_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row13007_block_007 : List Nat := []
def plan13007_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row13007_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 13007 row13007_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan13007_block_007
  · decide
  · rfl

def row13007_block_008 : List Nat := []
theorem row13007_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 13007 row13007_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row13007_block_009 : List Nat := []
theorem row13007_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 13007 row13007_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row13007_block_010 : List Nat := []
theorem row13007_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 13007 row13007_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row13007_block_011 : List Nat := []
theorem row13007_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 13007 row13007_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row13007_block_012 : List Nat := []
theorem row13007_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 13007 row13007_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row13007_block_013 : List Nat := []
theorem row13007_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 13007 row13007_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row13007_large_left_left : List Nat := row13007_small ++ row13007_block_005
theorem row13007_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 13007 row13007_large_left_left :=
  Budget.RowUpperBound.branch row13007_small_covers row13007_block_005_covers

def row13007_large_left_right_right : List Nat := row13007_block_007 ++ row13007_block_008
theorem row13007_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 13007 row13007_large_left_right_right :=
  Budget.RowUpperBound.branch row13007_block_007_covers row13007_block_008_covers

def row13007_large_left_right : List Nat := row13007_block_006 ++ row13007_large_left_right_right
theorem row13007_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 13007 row13007_large_left_right :=
  Budget.RowUpperBound.branch row13007_block_006_covers row13007_large_left_right_right_covers

def row13007_large_left : List Nat := row13007_large_left_left ++ row13007_large_left_right
theorem row13007_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 13007 row13007_large_left :=
  Budget.RowUpperBound.branch row13007_large_left_left_covers row13007_large_left_right_covers

def row13007_large_right_left : List Nat := row13007_block_009 ++ row13007_block_010
theorem row13007_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 13007 row13007_large_right_left :=
  Budget.RowUpperBound.branch row13007_block_009_covers row13007_block_010_covers

def row13007_large_right_right_right : List Nat := row13007_block_012 ++ row13007_block_013
theorem row13007_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 13007 row13007_large_right_right_right :=
  Budget.RowUpperBound.branch row13007_block_012_covers row13007_block_013_covers

def row13007_large_right_right : List Nat := row13007_block_011 ++ row13007_large_right_right_right
theorem row13007_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 13007 row13007_large_right_right :=
  Budget.RowUpperBound.branch row13007_block_011_covers row13007_large_right_right_right_covers

def row13007_large_right : List Nat := row13007_large_right_left ++ row13007_large_right_right
theorem row13007_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 13007 row13007_large_right :=
  Budget.RowUpperBound.branch row13007_large_right_left_covers row13007_large_right_right_covers

def row13007_large : List Nat := row13007_large_left ++ row13007_large_right
theorem row13007_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 13007 row13007_large :=
  Budget.RowUpperBound.branch row13007_large_left_covers row13007_large_right_covers

def row13007 : List Nat := row13007_large
theorem row13007_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 13007 row13007 := row13007_large_covers

end ProjectEulerSolutions.PrimePairLarge
