# Read one integer and check divisibility by 3 using subtraction
let n = input()

if n == 0
    print "Zero"
else
    if n > 0
        let x = n

        while x > 0
            x = x - 3
        end

        if x == 0
            print "Divisible by 3"
        else
            print "Not divisible by 3"
        end
    else
        print "Negative"
    end
end

