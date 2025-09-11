using  I from '../db/schema';

service AddressService {
  entity Addresses as projection on I.ADDRESS_2;

  action findClosest(prompt : String) returns Addresses;
}
