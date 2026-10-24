import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row12517_small : List Nat := []
theorem row12517_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12517 row12517_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12517_block_005 : List Nat := []
theorem row12517_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12517 row12517_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12517_block_006 : List Nat := []
theorem row12517_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12517 row12517_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12517_block_007 : List Nat := []
def plan12517_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .second, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 19), .first, .first, (.firstFactor 31), .second, (.secondFactor 23), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 17), (.secondFactor 13), .first, .first, .first, .second, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 23), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 29), (.firstFactor 13), (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12517_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12517 row12517_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12517_block_007
  · decide
  · rfl

def row12517_block_008 : List Nat := []
theorem row12517_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12517 row12517_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12517_block_009 : List Nat := []
theorem row12517_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12517 row12517_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12517_block_010 : List Nat := []
theorem row12517_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12517 row12517_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12517_block_011 : List Nat := []
theorem row12517_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12517 row12517_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12517_block_012 : List Nat := []
theorem row12517_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12517 row12517_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12517_block_013 : List Nat := []
theorem row12517_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12517 row12517_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12517_large_left_left : List Nat := row12517_small ++ row12517_block_005
theorem row12517_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12517 row12517_large_left_left :=
  Budget.RowUpperBound.branch row12517_small_covers row12517_block_005_covers

def row12517_large_left_right_right : List Nat := row12517_block_007 ++ row12517_block_008
theorem row12517_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12517 row12517_large_left_right_right :=
  Budget.RowUpperBound.branch row12517_block_007_covers row12517_block_008_covers

def row12517_large_left_right : List Nat := row12517_block_006 ++ row12517_large_left_right_right
theorem row12517_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12517 row12517_large_left_right :=
  Budget.RowUpperBound.branch row12517_block_006_covers row12517_large_left_right_right_covers

def row12517_large_left : List Nat := row12517_large_left_left ++ row12517_large_left_right
theorem row12517_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12517 row12517_large_left :=
  Budget.RowUpperBound.branch row12517_large_left_left_covers row12517_large_left_right_covers

def row12517_large_right_left : List Nat := row12517_block_009 ++ row12517_block_010
theorem row12517_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12517 row12517_large_right_left :=
  Budget.RowUpperBound.branch row12517_block_009_covers row12517_block_010_covers

def row12517_large_right_right_right : List Nat := row12517_block_012 ++ row12517_block_013
theorem row12517_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12517 row12517_large_right_right_right :=
  Budget.RowUpperBound.branch row12517_block_012_covers row12517_block_013_covers

def row12517_large_right_right : List Nat := row12517_block_011 ++ row12517_large_right_right_right
theorem row12517_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12517 row12517_large_right_right :=
  Budget.RowUpperBound.branch row12517_block_011_covers row12517_large_right_right_right_covers

def row12517_large_right : List Nat := row12517_large_right_left ++ row12517_large_right_right
theorem row12517_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12517 row12517_large_right :=
  Budget.RowUpperBound.branch row12517_large_right_left_covers row12517_large_right_right_covers

def row12517_large : List Nat := row12517_large_left ++ row12517_large_right
theorem row12517_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12517 row12517_large :=
  Budget.RowUpperBound.branch row12517_large_left_covers row12517_large_right_covers

def row12517 : List Nat := row12517_large
theorem row12517_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12517 row12517 := row12517_large_covers

def row12527_small : List Nat := []
theorem row12527_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12527 row12527_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12527_block_005 : List Nat := []
theorem row12527_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12527 row12527_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12527_block_006 : List Nat := []
theorem row12527_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12527 row12527_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12527_block_007 : List Nat := []
def plan12527_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 13), .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 23), (.secondFactor 13), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .second, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 23), (.firstFactor 31), .first, .first, .first, .first, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 23), (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 17), (.secondFactor 23), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12527_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12527 row12527_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12527_block_007
  · decide
  · rfl

def row12527_block_008 : List Nat := []
theorem row12527_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12527 row12527_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12527_block_009 : List Nat := []
theorem row12527_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12527 row12527_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12527_block_010 : List Nat := []
theorem row12527_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12527 row12527_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12527_block_011 : List Nat := []
theorem row12527_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12527 row12527_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12527_block_012 : List Nat := []
theorem row12527_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12527 row12527_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12527_block_013 : List Nat := []
theorem row12527_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12527 row12527_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12527_large_left_left : List Nat := row12527_small ++ row12527_block_005
theorem row12527_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12527 row12527_large_left_left :=
  Budget.RowUpperBound.branch row12527_small_covers row12527_block_005_covers

def row12527_large_left_right_right : List Nat := row12527_block_007 ++ row12527_block_008
theorem row12527_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12527 row12527_large_left_right_right :=
  Budget.RowUpperBound.branch row12527_block_007_covers row12527_block_008_covers

def row12527_large_left_right : List Nat := row12527_block_006 ++ row12527_large_left_right_right
theorem row12527_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12527 row12527_large_left_right :=
  Budget.RowUpperBound.branch row12527_block_006_covers row12527_large_left_right_right_covers

def row12527_large_left : List Nat := row12527_large_left_left ++ row12527_large_left_right
theorem row12527_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12527 row12527_large_left :=
  Budget.RowUpperBound.branch row12527_large_left_left_covers row12527_large_left_right_covers

def row12527_large_right_left : List Nat := row12527_block_009 ++ row12527_block_010
theorem row12527_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12527 row12527_large_right_left :=
  Budget.RowUpperBound.branch row12527_block_009_covers row12527_block_010_covers

def row12527_large_right_right_right : List Nat := row12527_block_012 ++ row12527_block_013
theorem row12527_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12527 row12527_large_right_right_right :=
  Budget.RowUpperBound.branch row12527_block_012_covers row12527_block_013_covers

def row12527_large_right_right : List Nat := row12527_block_011 ++ row12527_large_right_right_right
theorem row12527_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12527 row12527_large_right_right :=
  Budget.RowUpperBound.branch row12527_block_011_covers row12527_large_right_right_right_covers

def row12527_large_right : List Nat := row12527_large_right_left ++ row12527_large_right_right
theorem row12527_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12527 row12527_large_right :=
  Budget.RowUpperBound.branch row12527_large_right_left_covers row12527_large_right_right_covers

def row12527_large : List Nat := row12527_large_left ++ row12527_large_right
theorem row12527_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12527 row12527_large :=
  Budget.RowUpperBound.branch row12527_large_left_covers row12527_large_right_covers

def row12527 : List Nat := row12527_large
theorem row12527_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12527 row12527 := row12527_large_covers

def row12539_small : List Nat := []
theorem row12539_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12539 row12539_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12539_block_005 : List Nat := []
theorem row12539_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12539 row12539_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12539_block_006 : List Nat := []
theorem row12539_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12539 row12539_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12539_block_007 : List Nat := [12641, 13337]
def plan12539_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .possible, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 19), .first, .first, .first, .second, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 17), (.secondFactor 13), .first, (.firstFactor 19), (.secondFactor 17), (.secondFactor 37), (.firstFactor 13), .first, .first, .first, .second, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 17), (.firstFactor 29), .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 19), .first, .second, .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 11), .possible, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .second, (.firstFactor 11), (.secondFactor 7), .first, .second, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12539_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12539 row12539_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12539_block_007
  · decide
  · rfl

def row12539_block_008 : List Nat := []
theorem row12539_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12539 row12539_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12539_block_009 : List Nat := []
theorem row12539_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12539 row12539_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12539_block_010 : List Nat := []
theorem row12539_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12539 row12539_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12539_block_011 : List Nat := []
theorem row12539_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12539 row12539_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12539_block_012 : List Nat := []
theorem row12539_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12539 row12539_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12539_block_013 : List Nat := []
theorem row12539_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12539 row12539_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12539_large_left_left : List Nat := row12539_small ++ row12539_block_005
theorem row12539_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12539 row12539_large_left_left :=
  Budget.RowUpperBound.branch row12539_small_covers row12539_block_005_covers

def row12539_large_left_right_right : List Nat := row12539_block_007 ++ row12539_block_008
theorem row12539_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12539 row12539_large_left_right_right :=
  Budget.RowUpperBound.branch row12539_block_007_covers row12539_block_008_covers

def row12539_large_left_right : List Nat := row12539_block_006 ++ row12539_large_left_right_right
theorem row12539_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12539 row12539_large_left_right :=
  Budget.RowUpperBound.branch row12539_block_006_covers row12539_large_left_right_right_covers

def row12539_large_left : List Nat := row12539_large_left_left ++ row12539_large_left_right
theorem row12539_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12539 row12539_large_left :=
  Budget.RowUpperBound.branch row12539_large_left_left_covers row12539_large_left_right_covers

def row12539_large_right_left : List Nat := row12539_block_009 ++ row12539_block_010
theorem row12539_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12539 row12539_large_right_left :=
  Budget.RowUpperBound.branch row12539_block_009_covers row12539_block_010_covers

def row12539_large_right_right_right : List Nat := row12539_block_012 ++ row12539_block_013
theorem row12539_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12539 row12539_large_right_right_right :=
  Budget.RowUpperBound.branch row12539_block_012_covers row12539_block_013_covers

def row12539_large_right_right : List Nat := row12539_block_011 ++ row12539_large_right_right_right
theorem row12539_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12539 row12539_large_right_right :=
  Budget.RowUpperBound.branch row12539_block_011_covers row12539_large_right_right_right_covers

def row12539_large_right : List Nat := row12539_large_right_left ++ row12539_large_right_right
theorem row12539_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12539 row12539_large_right :=
  Budget.RowUpperBound.branch row12539_large_right_left_covers row12539_large_right_right_covers

def row12539_large : List Nat := row12539_large_left ++ row12539_large_right
theorem row12539_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12539 row12539_large :=
  Budget.RowUpperBound.branch row12539_large_left_covers row12539_large_right_covers

def row12539 : List Nat := row12539_large
theorem row12539_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12539 row12539 := row12539_large_covers

def row12541_small : List Nat := []
theorem row12541_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12541 row12541_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12541_block_005 : List Nat := []
theorem row12541_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12541 row12541_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12541_block_006 : List Nat := []
theorem row12541_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12541 row12541_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12541_block_007 : List Nat := []
def plan12541_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.secondFactor 7), .first, .second, .second, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .second, (.firstFactor 17), (.firstFactor 31), .first, (.secondFactor 17), .first, .first, .first, .second, (.firstFactor 19), (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 19), .first, .first, .second, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 23), (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .second, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 19), .first, .first, .second, .first, .first, (.firstFactor 23), .first, (.firstFactor 11), .first, (.firstFactor 19), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 29), .first, (.firstFactor 7), (.firstFactor 19), (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12541_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12541 row12541_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12541_block_007
  · decide
  · rfl

def row12541_block_008 : List Nat := []
theorem row12541_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12541 row12541_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12541_block_009 : List Nat := []
theorem row12541_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12541 row12541_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12541_block_010 : List Nat := []
theorem row12541_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12541 row12541_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12541_block_011 : List Nat := []
theorem row12541_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12541 row12541_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12541_block_012 : List Nat := []
theorem row12541_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12541 row12541_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12541_block_013 : List Nat := []
theorem row12541_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12541 row12541_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12541_large_left_left : List Nat := row12541_small ++ row12541_block_005
theorem row12541_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12541 row12541_large_left_left :=
  Budget.RowUpperBound.branch row12541_small_covers row12541_block_005_covers

def row12541_large_left_right_right : List Nat := row12541_block_007 ++ row12541_block_008
theorem row12541_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12541 row12541_large_left_right_right :=
  Budget.RowUpperBound.branch row12541_block_007_covers row12541_block_008_covers

def row12541_large_left_right : List Nat := row12541_block_006 ++ row12541_large_left_right_right
theorem row12541_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12541 row12541_large_left_right :=
  Budget.RowUpperBound.branch row12541_block_006_covers row12541_large_left_right_right_covers

def row12541_large_left : List Nat := row12541_large_left_left ++ row12541_large_left_right
theorem row12541_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12541 row12541_large_left :=
  Budget.RowUpperBound.branch row12541_large_left_left_covers row12541_large_left_right_covers

def row12541_large_right_left : List Nat := row12541_block_009 ++ row12541_block_010
theorem row12541_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12541 row12541_large_right_left :=
  Budget.RowUpperBound.branch row12541_block_009_covers row12541_block_010_covers

def row12541_large_right_right_right : List Nat := row12541_block_012 ++ row12541_block_013
theorem row12541_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12541 row12541_large_right_right_right :=
  Budget.RowUpperBound.branch row12541_block_012_covers row12541_block_013_covers

def row12541_large_right_right : List Nat := row12541_block_011 ++ row12541_large_right_right_right
theorem row12541_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12541 row12541_large_right_right :=
  Budget.RowUpperBound.branch row12541_block_011_covers row12541_large_right_right_right_covers

def row12541_large_right : List Nat := row12541_large_right_left ++ row12541_large_right_right
theorem row12541_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12541 row12541_large_right :=
  Budget.RowUpperBound.branch row12541_large_right_left_covers row12541_large_right_right_covers

def row12541_large : List Nat := row12541_large_left ++ row12541_large_right
theorem row12541_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12541 row12541_large :=
  Budget.RowUpperBound.branch row12541_large_left_covers row12541_large_right_covers

def row12541 : List Nat := row12541_large
theorem row12541_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12541 row12541 := row12541_large_covers

end ProjectEulerSolutions.PrimePairLarge
