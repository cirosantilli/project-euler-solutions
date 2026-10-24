import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row12547_small : List Nat := []
theorem row12547_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12547 row12547_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12547_block_005 : List Nat := []
theorem row12547_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12547 row12547_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12547_block_006 : List Nat := []
theorem row12547_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12547 row12547_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12547_block_007 : List Nat := [13099]
def plan12547_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .second, .first, (.secondFactor 7), (.firstFactor 23), (.firstFactor 13), (.secondFactor 19), .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 29), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 37), .first, .first, (.firstFactor 7), (.firstFactor 17), (.secondFactor 29), .second, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .second, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 13), .first, .first, (.firstFactor 31), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .second, .possible, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .second, .first, .first, .second, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 23), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 19), (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .second, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12547_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12547 row12547_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12547_block_007
  · decide
  · rfl

def row12547_block_008 : List Nat := []
theorem row12547_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12547 row12547_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12547_block_009 : List Nat := []
theorem row12547_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12547 row12547_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12547_block_010 : List Nat := []
theorem row12547_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12547 row12547_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12547_block_011 : List Nat := []
theorem row12547_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12547 row12547_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12547_block_012 : List Nat := []
theorem row12547_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12547 row12547_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12547_block_013 : List Nat := []
theorem row12547_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12547 row12547_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12547_large_left_left : List Nat := row12547_small ++ row12547_block_005
theorem row12547_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12547 row12547_large_left_left :=
  Budget.RowUpperBound.branch row12547_small_covers row12547_block_005_covers

def row12547_large_left_right_right : List Nat := row12547_block_007 ++ row12547_block_008
theorem row12547_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12547 row12547_large_left_right_right :=
  Budget.RowUpperBound.branch row12547_block_007_covers row12547_block_008_covers

def row12547_large_left_right : List Nat := row12547_block_006 ++ row12547_large_left_right_right
theorem row12547_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12547 row12547_large_left_right :=
  Budget.RowUpperBound.branch row12547_block_006_covers row12547_large_left_right_right_covers

def row12547_large_left : List Nat := row12547_large_left_left ++ row12547_large_left_right
theorem row12547_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12547 row12547_large_left :=
  Budget.RowUpperBound.branch row12547_large_left_left_covers row12547_large_left_right_covers

def row12547_large_right_left : List Nat := row12547_block_009 ++ row12547_block_010
theorem row12547_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12547 row12547_large_right_left :=
  Budget.RowUpperBound.branch row12547_block_009_covers row12547_block_010_covers

def row12547_large_right_right_right : List Nat := row12547_block_012 ++ row12547_block_013
theorem row12547_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12547 row12547_large_right_right_right :=
  Budget.RowUpperBound.branch row12547_block_012_covers row12547_block_013_covers

def row12547_large_right_right : List Nat := row12547_block_011 ++ row12547_large_right_right_right
theorem row12547_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12547 row12547_large_right_right :=
  Budget.RowUpperBound.branch row12547_block_011_covers row12547_large_right_right_right_covers

def row12547_large_right : List Nat := row12547_large_right_left ++ row12547_large_right_right
theorem row12547_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12547 row12547_large_right :=
  Budget.RowUpperBound.branch row12547_large_right_left_covers row12547_large_right_right_covers

def row12547_large : List Nat := row12547_large_left ++ row12547_large_right
theorem row12547_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12547 row12547_large :=
  Budget.RowUpperBound.branch row12547_large_left_covers row12547_large_right_covers

def row12547 : List Nat := row12547_large
theorem row12547_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12547 row12547 := row12547_large_covers

def row12553_small : List Nat := []
theorem row12553_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12553 row12553_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12553_block_005 : List Nat := []
theorem row12553_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12553 row12553_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12553_block_006 : List Nat := []
theorem row12553_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12553 row12553_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12553_block_007 : List Nat := []
def plan12553_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 19), (.firstFactor 7), (.firstFactor 17), (.secondFactor 7), .first, .first, (.firstFactor 11), (.secondFactor 19), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 37), .first, .first, (.firstFactor 29), (.secondFactor 23), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), (.secondFactor 19), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 37), .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 31), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), .second, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12553_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12553 row12553_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12553_block_007
  · decide
  · rfl

def row12553_block_008 : List Nat := []
theorem row12553_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12553 row12553_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12553_block_009 : List Nat := []
theorem row12553_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12553 row12553_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12553_block_010 : List Nat := []
theorem row12553_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12553 row12553_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12553_block_011 : List Nat := []
theorem row12553_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12553 row12553_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12553_block_012 : List Nat := []
theorem row12553_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12553 row12553_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12553_block_013 : List Nat := []
theorem row12553_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12553 row12553_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12553_large_left_left : List Nat := row12553_small ++ row12553_block_005
theorem row12553_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12553 row12553_large_left_left :=
  Budget.RowUpperBound.branch row12553_small_covers row12553_block_005_covers

def row12553_large_left_right_right : List Nat := row12553_block_007 ++ row12553_block_008
theorem row12553_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12553 row12553_large_left_right_right :=
  Budget.RowUpperBound.branch row12553_block_007_covers row12553_block_008_covers

def row12553_large_left_right : List Nat := row12553_block_006 ++ row12553_large_left_right_right
theorem row12553_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12553 row12553_large_left_right :=
  Budget.RowUpperBound.branch row12553_block_006_covers row12553_large_left_right_right_covers

def row12553_large_left : List Nat := row12553_large_left_left ++ row12553_large_left_right
theorem row12553_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12553 row12553_large_left :=
  Budget.RowUpperBound.branch row12553_large_left_left_covers row12553_large_left_right_covers

def row12553_large_right_left : List Nat := row12553_block_009 ++ row12553_block_010
theorem row12553_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12553 row12553_large_right_left :=
  Budget.RowUpperBound.branch row12553_block_009_covers row12553_block_010_covers

def row12553_large_right_right_right : List Nat := row12553_block_012 ++ row12553_block_013
theorem row12553_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12553 row12553_large_right_right_right :=
  Budget.RowUpperBound.branch row12553_block_012_covers row12553_block_013_covers

def row12553_large_right_right : List Nat := row12553_block_011 ++ row12553_large_right_right_right
theorem row12553_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12553 row12553_large_right_right :=
  Budget.RowUpperBound.branch row12553_block_011_covers row12553_large_right_right_right_covers

def row12553_large_right : List Nat := row12553_large_right_left ++ row12553_large_right_right
theorem row12553_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12553 row12553_large_right :=
  Budget.RowUpperBound.branch row12553_large_right_left_covers row12553_large_right_right_covers

def row12553_large : List Nat := row12553_large_left ++ row12553_large_right
theorem row12553_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12553 row12553_large :=
  Budget.RowUpperBound.branch row12553_large_left_covers row12553_large_right_covers

def row12553 : List Nat := row12553_large
theorem row12553_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12553 row12553 := row12553_large_covers

def row12569_small : List Nat := []
theorem row12569_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12569 row12569_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12569_block_005 : List Nat := []
theorem row12569_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12569 row12569_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12569_block_006 : List Nat := []
theorem row12569_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12569 row12569_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12569_block_007 : List Nat := [12611, 12659]
def plan12569_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .first, .possible, .first, .first, .first, .first, (.secondFactor 17), (.secondFactor 37), .possible, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .second, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 23), .second, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 17), (.secondFactor 7), .first, .first, .first, (.firstFactor 29), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .second, (.secondFactor 29), (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 29), (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 13), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12569_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12569 row12569_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12569_block_007
  · decide
  · rfl

def row12569_block_008 : List Nat := []
theorem row12569_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12569 row12569_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12569_block_009 : List Nat := []
theorem row12569_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12569 row12569_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12569_block_010 : List Nat := []
theorem row12569_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12569 row12569_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12569_block_011 : List Nat := []
theorem row12569_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12569 row12569_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12569_block_012 : List Nat := []
theorem row12569_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12569 row12569_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12569_block_013 : List Nat := []
theorem row12569_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12569 row12569_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12569_large_left_left : List Nat := row12569_small ++ row12569_block_005
theorem row12569_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12569 row12569_large_left_left :=
  Budget.RowUpperBound.branch row12569_small_covers row12569_block_005_covers

def row12569_large_left_right_right : List Nat := row12569_block_007 ++ row12569_block_008
theorem row12569_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12569 row12569_large_left_right_right :=
  Budget.RowUpperBound.branch row12569_block_007_covers row12569_block_008_covers

def row12569_large_left_right : List Nat := row12569_block_006 ++ row12569_large_left_right_right
theorem row12569_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12569 row12569_large_left_right :=
  Budget.RowUpperBound.branch row12569_block_006_covers row12569_large_left_right_right_covers

def row12569_large_left : List Nat := row12569_large_left_left ++ row12569_large_left_right
theorem row12569_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12569 row12569_large_left :=
  Budget.RowUpperBound.branch row12569_large_left_left_covers row12569_large_left_right_covers

def row12569_large_right_left : List Nat := row12569_block_009 ++ row12569_block_010
theorem row12569_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12569 row12569_large_right_left :=
  Budget.RowUpperBound.branch row12569_block_009_covers row12569_block_010_covers

def row12569_large_right_right_right : List Nat := row12569_block_012 ++ row12569_block_013
theorem row12569_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12569 row12569_large_right_right_right :=
  Budget.RowUpperBound.branch row12569_block_012_covers row12569_block_013_covers

def row12569_large_right_right : List Nat := row12569_block_011 ++ row12569_large_right_right_right
theorem row12569_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12569 row12569_large_right_right :=
  Budget.RowUpperBound.branch row12569_block_011_covers row12569_large_right_right_right_covers

def row12569_large_right : List Nat := row12569_large_right_left ++ row12569_large_right_right
theorem row12569_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12569 row12569_large_right :=
  Budget.RowUpperBound.branch row12569_large_right_left_covers row12569_large_right_right_covers

def row12569_large : List Nat := row12569_large_left ++ row12569_large_right
theorem row12569_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12569 row12569_large :=
  Budget.RowUpperBound.branch row12569_large_left_covers row12569_large_right_covers

def row12569 : List Nat := row12569_large
theorem row12569_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12569 row12569 := row12569_large_covers

def row12577_small : List Nat := []
theorem row12577_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12577 row12577_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12577_block_005 : List Nat := []
theorem row12577_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12577 row12577_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12577_block_006 : List Nat := []
theorem row12577_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12577 row12577_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12577_block_007 : List Nat := []
def plan12577_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 7), .second, .first, (.secondFactor 7), (.firstFactor 19), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, (.secondFactor 37), (.secondFactor 7), .first, (.firstFactor 7), .second, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 23), .first, .first, (.secondFactor 31), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.secondFactor 23), (.firstFactor 7), (.secondFactor 17), .first, .first, .first, .first, .second, .first, (.firstFactor 17), .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 31), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 29), (.secondFactor 13), .second, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 23), .first, (.secondFactor 29), .first, .first, (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 29), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12577_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12577 row12577_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12577_block_007
  · decide
  · rfl

def row12577_block_008 : List Nat := []
theorem row12577_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12577 row12577_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12577_block_009 : List Nat := []
theorem row12577_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12577 row12577_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12577_block_010 : List Nat := []
theorem row12577_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12577 row12577_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12577_block_011 : List Nat := []
theorem row12577_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12577 row12577_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12577_block_012 : List Nat := []
theorem row12577_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12577 row12577_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12577_block_013 : List Nat := []
theorem row12577_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12577 row12577_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12577_large_left_left : List Nat := row12577_small ++ row12577_block_005
theorem row12577_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12577 row12577_large_left_left :=
  Budget.RowUpperBound.branch row12577_small_covers row12577_block_005_covers

def row12577_large_left_right_right : List Nat := row12577_block_007 ++ row12577_block_008
theorem row12577_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12577 row12577_large_left_right_right :=
  Budget.RowUpperBound.branch row12577_block_007_covers row12577_block_008_covers

def row12577_large_left_right : List Nat := row12577_block_006 ++ row12577_large_left_right_right
theorem row12577_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12577 row12577_large_left_right :=
  Budget.RowUpperBound.branch row12577_block_006_covers row12577_large_left_right_right_covers

def row12577_large_left : List Nat := row12577_large_left_left ++ row12577_large_left_right
theorem row12577_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12577 row12577_large_left :=
  Budget.RowUpperBound.branch row12577_large_left_left_covers row12577_large_left_right_covers

def row12577_large_right_left : List Nat := row12577_block_009 ++ row12577_block_010
theorem row12577_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12577 row12577_large_right_left :=
  Budget.RowUpperBound.branch row12577_block_009_covers row12577_block_010_covers

def row12577_large_right_right_right : List Nat := row12577_block_012 ++ row12577_block_013
theorem row12577_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12577 row12577_large_right_right_right :=
  Budget.RowUpperBound.branch row12577_block_012_covers row12577_block_013_covers

def row12577_large_right_right : List Nat := row12577_block_011 ++ row12577_large_right_right_right
theorem row12577_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12577 row12577_large_right_right :=
  Budget.RowUpperBound.branch row12577_block_011_covers row12577_large_right_right_right_covers

def row12577_large_right : List Nat := row12577_large_right_left ++ row12577_large_right_right
theorem row12577_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12577 row12577_large_right :=
  Budget.RowUpperBound.branch row12577_large_right_left_covers row12577_large_right_right_covers

def row12577_large : List Nat := row12577_large_left ++ row12577_large_right
theorem row12577_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12577 row12577_large :=
  Budget.RowUpperBound.branch row12577_large_left_covers row12577_large_right_covers

def row12577 : List Nat := row12577_large
theorem row12577_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12577 row12577 := row12577_large_covers

end ProjectEulerSolutions.PrimePairLarge
